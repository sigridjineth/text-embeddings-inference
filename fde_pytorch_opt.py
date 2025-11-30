# fde_pytorch_opt.py
# MUVERA Fixed Dimensional Encodings (FDE) - Optimized + Sampling/Minibatch/Progressive Refinement
# - Batched SimHash (φ) & inner projection (ψ) across R reps (einsum)
# - Hypercube neighbor fill for empty buckets (avoid full Hamming matrix)
# - Streaming final projection (avoid giant concatenation peak memory)
# - Vectorized/streaming scoring
# - Optional mixed precision params
# - Sampling (token-level PPS / per-bucket cap / ratio) & minibatch doc processing
# - Progressive coarse→refine Top-K selection (R_sub + token subsampling)

from __future__ import annotations
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal

import torch
import torch.nn.functional as F

# ------------------------------
# Timing utilities
# ------------------------------
@contextmanager
def timer(label: str):
    _t0 = time.perf_counter()
    try:
        yield
    finally:
        _ms = (time.perf_counter() - _t0) # * 1000.0
        print(f"[TIME] {label}: {_ms:.2f} s")

# ------------------------------
# Utilities
# ------------------------------
def get_device() -> torch.device:
    # Uncomment to use CUDA/MPS if available
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    #return torch.device("cpu")

def normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def make_pm1_matrix(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    return (torch.randint(0, 2, (rows, cols), device=device, dtype=torch.int8) * 2 - 1).float()

# ------------------------------
# Batched parameters for R reps
# ------------------------------
class BatchedParams:
    """
    φ (SimHash directions) and ψ (inner projection) for all repetitions in batched form.
    G: (R, d, ksim)
    S: (R, d_proj, d) or None (identity)
    """
    def __init__(self, d: int, ksim: int, d_proj: Optional[int], R: int, device: torch.device, seed: int):
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        self.G = torch.randn(R, d, ksim, device=device, generator=g)  # Gaussian directions
        self.use_proj = bool(d_proj and d_proj > 0 and d_proj != d)
        if self.use_proj:
            self.S = (torch.randint(0, 2, (R, d_proj, d), device=device, dtype=torch.int8, generator=g) * 2 - 1).to(torch.float32)
        else:
            self.S = None

# ------------------------------
# Core batched kernels
# ------------------------------
def _buckets_batched(X: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """
    X: (m, d)
    G: (R, d, ksim)
    returns: bucket ids (R, m) in [0, 2^ksim)
    """
    logits = torch.einsum('md,Rdk->Rmk', X, G)   # (R,m,ksim)
    bits = (logits > 0).to(torch.int64)          # (R,m,ksim)
    powers = (1 << torch.arange(G.size(-1), device=X.device, dtype=torch.int64))
    return (bits * powers).sum(dim=-1)           # (R,m)

def _aggregate_doc_batched(X: torch.Tensor, bucket_ids: torch.Tensor, B: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    X: (m,d), bucket_ids: (R,m)
    returns sums: (R,B,d), counts: (R,B)
    """
    R, m = bucket_ids.shape
    d = X.size(1)
    sums = X.new_zeros((R, B, d))
    counts = X.new_zeros((R, B))
    for r in range(R):
        sums[r].index_add_(0, bucket_ids[r], X)
        counts[r] = torch.bincount(bucket_ids[r], minlength=B).to(X.dtype)
    return sums, counts

def _aggregate_query_batched(X: torch.Tensor, bucket_ids: torch.Tensor, B: int) -> torch.Tensor:
    R, m = bucket_ids.shape
    d = X.size(1)
    sums = X.new_zeros((R, B, d))
    for r in range(R):
        sums[r].index_add_(0, bucket_ids[r], X)
    return sums

def _project_block_batched(blocks: torch.Tensor, S: Optional[torch.Tensor]) -> torch.Tensor:
    """
    blocks: (R,B,d)
    S: (R,d_proj,d) or None
    returns: (R,B,d_proj) if S else (R,B,d)
    """
    if S is None:
        return blocks
    return torch.einsum('Rbd,Rkd->Rbk', blocks, S)

# ------------------------------
# Hypercube neighbor fill (empty buckets)
# ------------------------------
def _fill_empty_doc_buckets_hcube(centroids: torch.Tensor, counts: torch.Tensor, ksim: int) -> torch.Tensor:
    """
    centroids: (B, d), counts: (B,)
    For each empty bucket, search Hamming neighbors by radius 1 then 2 and copy a present centroid.
    """
    empty_mask = (counts == 0)
    if not empty_mask.any():
        return centroids

    B, d = centroids.shape
    present = ~empty_mask
    if present.sum() == 0:
        return centroids  # all empty

    masks = (1 << torch.arange(ksim, device=centroids.device, dtype=torch.int64))  # (ksim,)
    empty_idx = torch.nonzero(empty_mask, as_tuple=False).squeeze(-1)

    for k in empty_idx.tolist():
        # radius 1
        n1 = (k ^ masks).clamp_(min=0, max=B-1)   # neighbors flipping one bit
        cand = n1[present[n1]]
        if cand.numel() > 0:
            centroids[k] = centroids[cand[0]]
            continue
        # radius 2
        found = False
        for i in range(ksim):
            if found: break
            for j in range(i+1, ksim):
                nei = k ^ masks[i] ^ masks[j]
                if 0 <= nei < B and present[nei]:
                    centroids[k] = centroids[nei]
                    found = True
                    break
        # else remain zeros
    return centroids

# ------------------------------
# Streaming final projection
# ------------------------------
class FinalProjectionStreamer:
    """
    y = W @ x without materializing full x when x is many concatenated blocks.
    W: (d_out, d_in)
    add_slice(x_slice, col_start): y += W[:, col_start:col_start+len(x_slice)] @ x_slice
    """
    def __init__(self, d_in: int, d_out: int, device: torch.device):
        assert d_out > 0 and d_out <= d_in
        self.W = torch.randn(d_out, d_in, device=device)
        self.d_out = d_out
        self.d_in = d_in

    def zero(self) -> torch.Tensor:
        return torch.zeros(self.d_out, device=self.W.device)

    def add_slice(self, y: torch.Tensor, x_slice: torch.Tensor, col_start: int) -> torch.Tensor:
        cols = x_slice.numel()
        Wsub = self.W[:, col_start:col_start + cols]  # (d_out, cols)
        return y.add_(Wsub @ x_slice)

# ------------------------------
# FDE Builder (optimized)
# ------------------------------
class FDEConfig:
    def __init__(
        self,
        ksim: int = 6,
        d_proj: int = 32,
        R_reps: int = 10,
        d_final: Optional[int] = None,
        fill_empty_clusters: bool = True,
        seed: int = 42,
        use_mixed_precision: bool = False,  # params in fp16/bf16, compute in fp32
    ):
        self.ksim = ksim
        self.d_proj = d_proj
        self.R_reps = R_reps
        self.d_final = d_final
        self.fill_empty_clusters = fill_empty_clusters
        self.seed = seed
        self.use_mixed_precision = use_mixed_precision

class FDEBuilder:
    """
    Build FDEs for documents and queries (batched φ/ψ, hypercube fill, streaming final proj).
    """
    def __init__(self, d: int, config: FDEConfig, device: Optional[torch.device] = None):
        self.d = d
        self.cfg = config
        self.device = device or get_device()

        self.R = self.cfg.R_reps
        self.B = 1 << self.cfg.ksim

        self.params = BatchedParams(d, self.cfg.ksim, self.cfg.d_proj, self.R, self.device, self.cfg.seed)

        d_block = self.cfg.d_proj if (self.params.use_proj and self.cfg.d_proj) else d
        self.d_block = d_block
        self.d_FDE = self.B * d_block * self.R

        if self.cfg.d_final:
            self.final_proj = FinalProjectionStreamer(self.d_FDE, self.cfg.d_final, self.device)
        else:
            self.final_proj = None

        if self.cfg.use_mixed_precision:
            if self.params.S is not None:
                self.params.S = self.params.S.to(torch.float16)
            self.params.G = self.params.G.to(torch.float16)
            if self.final_proj:
                self.final_proj.W = self.final_proj.W.to(torch.float16)

    def _fill_empty_doc_buckets_batched(self, centroids: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        if not self.cfg.fill_empty_clusters:
            return centroids
        R, B, d = centroids.shape
        for r in range(R):
            centroids[r] = _fill_empty_doc_buckets_hcube(centroids[r], counts[r], self.cfg.ksim)
        return centroids

    @torch.no_grad()
    def build_document_fde(self, X: torch.Tensor) -> torch.Tensor:
        assert X.dim() == 2 and X.size(1) == self.d
        X = normalize_rows(X)

        # ← 활성 R만 사용
        R_use = self.R

        # G, S를 먼저 slice한 뒤 필요시 캐스팅 (불필요한 전체 R 캐스팅/계산 방지)
        G = self.params.G[:R_use]
        if self.cfg.use_mixed_precision:
            G = G.to(torch.float32)

        bid = _buckets_batched(X, G)                                  # (R_use, m)

        sums, counts = _aggregate_doc_batched(X, bid, self.B)         # (R_use, B, d), (R_use, B)
        counts_safe = counts.clone()
        counts_safe[counts_safe == 0] = 1.0
        centroids = sums / counts_safe.unsqueeze(-1)                  # (R_use, B, d)

        if self.cfg.fill_empty_clusters:
            centroids = self._fill_empty_doc_buckets_batched(centroids, counts)

        S = None
        if self.params.S is not None:
            S = self.params.S[:R_use]
            if self.cfg.use_mixed_precision:
                S = S.to(torch.float32)

        blocks = _project_block_batched(centroids, S)                 # (R_use, B, d_block)

        if self.final_proj:
            if self.cfg.use_mixed_precision:
                self.final_proj.W = self.final_proj.W.to(torch.float32)
            y = self.final_proj.zero()
            col = 0
            for r in range(R_use):
                flat = blocks[r].reshape(-1)                          # (B * d_block,)
                y = self.final_proj.add_slice(y, flat, col)
                col += flat.numel()
            return y
        else:
            return blocks.reshape(-1)                                  # (R_use * B * d_block,)


    @torch.no_grad()
    def build_query_fde(self, X: torch.Tensor) -> torch.Tensor:
        assert X.dim() == 2 and X.size(1) == self.d
        X = normalize_rows(X)

        R_use = self.R

        G = self.params.G[:R_use]
        if self.cfg.use_mixed_precision:
            G = G.to(torch.float32)

        bid = _buckets_batched(X, G)                                  # (R_use, m)

        sums = _aggregate_query_batched(X, bid, self.B)               # (R_use, B, d)

        S = None
        if self.params.S is not None:
            S = self.params.S[:R_use]
            if self.cfg.use_mixed_precision:
                S = S.to(torch.float32)

        blocks = _project_block_batched(sums, S)                      # (R_use, B, d_block)

        if self.final_proj:
            if self.cfg.use_mixed_precision:
                self.final_proj.W = self.final_proj.W.to(torch.float32)
            y = self.final_proj.zero()
            col = 0
            for r in range(R_use):
                flat = blocks[r].reshape(-1)
                y = self.final_proj.add_slice(y, flat, col)
                col += flat.numel()
            return y
        else:
            return blocks.reshape(-1)

# ------------------------------
# Chamfer similarity (MaxSim)
# ------------------------------
@torch.no_grad()
def chamfer_similarity(query_tokens: torch.Tensor, doc_tokens: torch.Tensor) -> float:
    q = normalize_rows(query_tokens)
    p = normalize_rows(doc_tokens)
    sims = q @ p.t()
    max_per_q, _ = sims.max(dim=1)
    return max_per_q.sum().item()

# ------------------------------
# Optional int8 quantization (storage/fast dot)
# ------------------------------
@torch.no_grad()
def quantize_int8(x: torch.Tensor, eps: float = 1e-8):
    s = x.abs().max().clamp(min=eps)
    q = (x / s * 127.0).round().to(torch.int8)
    return q, s

@torch.no_grad()
def dot_int8(q_int8: torch.Tensor, s_q: torch.Tensor, d_int8: torch.Tensor, s_d: torch.Tensor) -> float:
    return (q_int8.to(torch.int32) @ d_int8.to(torch.int32)).item() * float((s_q * s_d) / (127.0 * 127.0))

# ------------------------------
# Scoring utilities
# ------------------------------
@torch.no_grad()
def dot_stream(q: torch.Tensor, docs: List[torch.Tensor], chunk: int = 65536) -> torch.Tensor:
    out = []
    i = 0
    while i < len(docs):
        slab = torch.stack(docs[i:i + chunk], 0)  # (c, D)
        out.append(slab @ q)
        i += chunk
    return torch.cat(out, 0)

# ------------------------------
# Sampling & Minibatch
# ------------------------------
@dataclass
class SamplerConfig:
    token_sample_ratio: float = 1.0   # e.g., 0.2 for 20% of tokens
    per_bucket_cap: int = 0           # 0 = no cap; else max tokens per bucket
    use_pps: bool = False             # PPS by token norm (or other weight)
    r_sub: int = 0                    # use only first r_sub reps for coarse pass (0 = all)
    seed: int = 123

def sample_tokens(X: torch.Tensor, sampler: SamplerConfig, builder: 'FDEBuilder') -> torch.Tensor:
    """
    Subsample rows of X according to SamplerConfig.
    If per_bucket_cap>0, bucketize (by first rep) then cap per bucket.
    If token_sample_ratio<1.0, apply uniform or PPS sampling.
    """
    torch.manual_seed(sampler.seed)
    m, d = X.shape
    if sampler.token_sample_ratio >= 1.0 and sampler.per_bucket_cap <= 0:
        return X

    # PPS weights (based on row L2 norm)
    p = None
    if sampler.use_pps:
        w = X.norm(dim=1) + 1e-9
        p = (w / w.sum()).clamp_min_(1e-12)

    # Per-bucket cap flow
    use_cap = sampler.per_bucket_cap > 0
    if use_cap:
        R_use = builder.R if sampler.r_sub <= 0 else min(sampler.r_sub, builder.R)
        G0 = builder.params.G[0:1] if not builder.cfg.use_mixed_precision else builder.params.G[0:1].to(torch.float32)
        bid = _buckets_batched(normalize_rows(X), G0)[0]  # (m,)
        B = builder.B
        selected_idx = []
        for b in range(B):
            idx = torch.nonzero(bid == b, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            # preliminary downsample by ratio
            if sampler.token_sample_ratio < 1.0:
                k = max(1, int(idx.numel() * sampler.token_sample_ratio))
            else:
                k = idx.numel()
            if p is not None:
                p_b = p[idx]
                p_b = p_b / p_b.sum()
                k = min(k, sampler.per_bucket_cap)
                choice = torch.multinomial(p_b, num_samples=min(k, idx.numel()), replacement=False)
                chosen = idx[choice]
            else:
                chosen = idx[torch.randperm(idx.numel())[:min(k, idx.numel())]]
                if chosen.numel() > sampler.per_bucket_cap:
                    chosen = chosen[:sampler.per_bucket_cap]
            selected_idx.append(chosen)
        if len(selected_idx) == 0:
            # fallback: uniform by ratio
            k = max(1, int(m * sampler.token_sample_ratio))
            choice = torch.randperm(m)[:min(k, m)]
            return X[choice]
        sel = torch.cat(selected_idx, dim=0)
        return X[sel.unique()]
    else:
        # plain uniform or PPS by global ratio
        k = max(1, int(m * sampler.token_sample_ratio))
        if p is not None:
            choice = torch.multinomial(p, num_samples=min(k, m), replacement=False)
        else:
            choice = torch.randperm(m)[:k]
        return X[choice]

@torch.no_grad()
def build_document_fde_batch(
    docs_tokens: List[torch.Tensor],
    builder: 'FDEBuilder',
    sampler: Optional[SamplerConfig] = None,
    batch_size_docs: int = 256,
    r_sub_override: int = 0,
) -> List[torch.Tensor]:
    """
    Build FDEs for many documents in minibatches to improve throughput.
    Optionally sample tokens inside each doc. Optionally limit R (coarse pass).
    """
    out: List[torch.Tensor] = []
    R_orig = builder.R
    if r_sub_override > 0 and r_sub_override < builder.R:
        builder.R = r_sub_override

    i = 0
    while i < len(docs_tokens):
        slab = docs_tokens[i:i + batch_size_docs]
        for D in slab:
            X = D
            if sampler is not None:
                X = sample_tokens(X, sampler, builder)
            out.append(builder.build_document_fde(X))
        i += batch_size_docs

    builder.R = R_orig
    return out

@torch.no_grad()
def progressive_score_topk(
    q_fde: torch.Tensor,
    docs_tokens: List[torch.Tensor],
    builder: 'FDEBuilder',
    k: int,
    coarse: SamplerConfig,
    refine: SamplerConfig,
    batch_size_docs: int = 256,
    candidate_factor: int = 20,
) -> Tuple[List[int], torch.Tensor]:
    """
    2-stage scoring:
      1) Coarse: limit repetitions (r_sub) + aggressive token subsampling -> pick top-M
      2) Refine: full repetitions + milder/no subsampling on M docs -> final top-k
    Returns (final_indices, final_scores)
    """
    M = min(len(docs_tokens), max(k, k * candidate_factor))
    coarse_R = coarse.r_sub if coarse.r_sub > 0 else max(1, builder.R // 4)

    with timer("Coarse FDE build (minibatch + sampling)"):
        doc_fdes_coarse = build_document_fde_batch(
            docs_tokens, builder, sampler=coarse, batch_size_docs=batch_size_docs, r_sub_override=coarse_R
        )
    with timer("Coarse scoring"):
        scores_coarse = dot_stream(q_fde, doc_fdes_coarse, chunk=65536)
    topM = torch.topk(scores_coarse, k=M).indices.tolist()

    cand_docs = [docs_tokens[i] for i in topM]
    with timer("Refine FDE build (minibatch + sampling)"):
        doc_fdes_ref = build_document_fde_batch(
            cand_docs, builder, sampler=refine, batch_size_docs=batch_size_docs, r_sub_override=0
        )
    with timer("Refine scoring"):
        scores_ref = dot_stream(q_fde, doc_fdes_ref, chunk=65536)

    topk_local = torch.topk(scores_ref, k=min(k, len(cand_docs)))
    final_idx = [topM[i] for i in topk_local.indices.tolist()]
    final_scores = topk_local.values
    return final_idx, final_scores

# ------------------------------
# Example end-to-end usage
# ------------------------------
def example():
    device = get_device()
    torch.manual_seed(0)

    # Toy dimensions
    d = 128                  # token embedding dim (ColBERT-style)
    m_q = 32                 # # query tokens
    m_d = 100                # # doc tokens
    num_docs = 100000         # increase to stress-test

    with timer("Generate random data (Q + D_tokens_list)"):
        Q_tokens = normalize_rows(torch.randn(m_q, d, device=device))
        D_tokens_list = [normalize_rows(torch.randn(m_d, d, device=device)) for _ in range(num_docs)]

    cfg = FDEConfig(
        ksim=5,             # B=32
        d_proj=16,          # per-bucket proj
        R_reps=10,          # repetitions
        d_final=1024,       # final dimension; set None to skip
        fill_empty_clusters=True,
        seed=123,
        use_mixed_precision=False,
    )

    with timer("Init FDEBuilder"):
        builder = FDEBuilder(d=d, config=cfg, device=device)

    # 서빙 파라메터 저장
    state = {
        "G": builder.params.G.detach().cpu(),
        "S": builder.params.S.detach().cpu() if builder.params.S is not None else None,
        "W": builder.final_proj.W.detach().cpu() if builder.final_proj is not None else None,
    }
    torch.save(state, "fde_params.pt")


    with timer("Build query FDE"):
        q_fde = builder.build_query_fde(Q_tokens)

    # --- Baseline full scoring (optional) ---
    # with timer("Build all doc FDEs (full)"):
    #     doc_fdes_full = [builder.build_document_fde(D) for D in D_tokens_list]
    # with timer("Full scoring"):
    #     scores_full = dot_stream(q_fde, doc_fdes_full, chunk=16384)
    # topk_full = torch.topk(scores_full, k=10)

    # --- Progressive 2-stage scoring with sampling/minibatch ---
    coarse = SamplerConfig(
        token_sample_ratio=0.25,   # use 25% tokens per doc
        per_bucket_cap=8,          # cap tokens per bucket
        use_pps=True,              # PPS by token norm
        r_sub=max(1, builder.R // 4),  # use subset of reps
        seed=123
    )
    refine = SamplerConfig(
        token_sample_ratio=0.8,    # richer sampling
        per_bucket_cap=0,          # no cap
        use_pps=False,             # uniform
        r_sub=0,                   # full reps
        seed=321
    )

    with timer("Progressive Top-K (coarse→refine)"):
        final_idx, final_scores = progressive_score_topk(
            q_fde, D_tokens_list, builder, k=10,
            coarse=coarse, refine=refine,
            batch_size_docs=256, candidate_factor=20
        )
    print("Final Top-10 doc indices:", final_idx)
    print("Final Top-10 scores:", final_scores.tolist())

    # Optional: Chamfer re-ranking on final top-K
    with timer("Chamfer re-ranking on final Top-K"):
        pairs = []
        for i in final_idx:
            score = chamfer_similarity(Q_tokens, D_tokens_list[i])
            pairs.append((i, score))
        pairs.sort(key=lambda x: x[1], reverse=True)
    print("Re-ranked by Chamfer (idx, score):", pairs)

    print(f"FDE dimension used: {builder.d_FDE if not builder.final_proj else cfg.d_final}")

if __name__ == "__main__":
    example()

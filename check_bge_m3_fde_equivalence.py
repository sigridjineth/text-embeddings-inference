#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BGE-M3 FDE Python vs TEI 동등성 체크 스크립트

- Python 측:
    - BGEM3FlagModel(return_colbert_vecs=True)로 colbert_vecs 추출
    - fde_pytorch_opt.FDEBuilder로 FDE 임베딩 생성

- TEI 측:
    - text-embeddings-router (pooling=bge_m3_fde) 에
      /v1/embeddings 엔드포인트로 HTTP POST

두 쪽 임베딩을 비교해서:
    - L2 distance
    - max |diff|
    - cosine similarity

를 출력하고, 전체 요약 통계를 보여줍니다.

예시 실행:

  # TEI 서버 실행
  text-embeddings-router \
    --model-id BAAI/bge-m3 \
    --pooling bge_m3_fde \
    --port 8080

  # 같은 쉘에서 (환경변수 동일하게):
  export FDE_KSIM=5
  export FDE_D_PROJ=16
  export FDE_R_REPS=10
  export FDE_D_FINAL=1024
  export FDE_SEED=42

  python check_bge_m3_fde_equivalence.py \
    --tei-url http://localhost:8080/v1/embeddings \
    --tei-model BAAI/bge-m3 \
    --bge-model BAAI/bge-m3 \
    --texts "hello world" "quick brown fox" "오늘 날씨 어때?"

"""

import argparse
import json
import os
from typing import List, Optional

import numpy as np
import requests
import torch

from FlagEmbedding import BGEM3FlagModel

from fde_pytorch_opt import (
    FDEConfig,
    FDEBuilder,
    normalize_rows,
)


# -------------------------------------------------------------------
# BGE-M3 ColBERT encoder (Python side)
# -------------------------------------------------------------------

class BGEColBERTEncoder:
    """
    BGE-M3 의 colbert_vecs 를 사용해서
    텍스트 -> 토큰 임베딩 리스트 [Tensor(N_i, d)] 로 변환
    """

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        use_fp16: bool = False,
        max_length: int = 512,
        batch_size: int = 16,
    ):
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        print(f"[PY] Loading BGEM3FlagModel: {model_name} (use_fp16={use_fp16})")
        # FlagEmbedding 쪽에서 내부적으로 device 선택
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

    def encode_texts(self, texts: List[str]) -> List[torch.Tensor]:
        """
        여러 문장을 한 번에 encode -> 리스트[Tensor(N_i, d)] 반환
        """
        all_tokens: List[torch.Tensor] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            out = self.model.encode(
                batch,
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=False,
                return_sparse=False,
                return_colbert_vecs=True,
            )
            colbert_vecs = out["colbert_vecs"]  # list of np.ndarray
            for arr in colbert_vecs:
                t = torch.tensor(arr, dtype=torch.float32, device=self.device)
                # 안전하게 한 번 더 L2 정규화
                t = normalize_rows(t)
                all_tokens.append(t)
        return all_tokens


# -------------------------------------------------------------------
# TEI 호출: /v1/embeddings (또는 /embed 호환)
# -------------------------------------------------------------------

def tei_embed(
    texts: List[str],
    tei_url: str,
    tei_model: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 60.0,
) -> np.ndarray:
    """
    TEI 서버에 HTTP POST로 임베딩 요청.

    기본적으로 /v1/embeddings (OpenAI 스타일)를 가정:
      POST tei_url
      JSON: {"model": "...", "input": [...]}

    만약 /embed 엔드포인트를 쓰면 응답이 {"embeddings": [...]} 일 수 있어
    data / embeddings 둘 다 처리.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {"input": texts}
    if tei_model is not None:
        payload["model"] = tei_model

    print(f"[TEI] POST {tei_url}")
    resp = requests.post(tei_url, headers=headers, data=json.dumps(payload), timeout=timeout)
    resp.raise_for_status()
    j = resp.json()

    if "data" in j:  # OpenAI /v1/embeddings 스타일
        embs = [item["embedding"] for item in j["data"]]
    elif "embeddings" in j:  # /embed 스타일
        embs = j["embeddings"]
    else:
        raise RuntimeError(f"Unexpected TEI response format: keys={list(j.keys())}")

    arr = np.asarray(embs, dtype=np.float32)
    print(f"[TEI] Got embeddings: shape={arr.shape}")
    return arr


# -------------------------------------------------------------------
# Python FDE: BGE + FDEBuilder
# -------------------------------------------------------------------

def build_fde_config_from_env_and_args(args) -> FDEConfig:
    """
    TEI와 동일한 설정을 쓰고 싶으면, 같은 환경변수(FDE_KSIM 등)를
    사용하거나 CLI 인자로 override 할 수 있음.
    """
    def _env_or_default(name: str, default: int) -> int:
        v = os.getenv(name)
        return int(v) if v is not None else default

    ksim = args.fde_ksim if args.fde_ksim is not None else _env_or_default("FDE_KSIM", 5)
    d_proj = args.fde_d_proj if args.fde_d_proj is not None else _env_or_default("FDE_D_PROJ", 16)
    r_reps = args.fde_r_reps if args.fde_r_reps is not None else _env_or_default("FDE_R_REPS", 10)

    d_final_env = os.getenv("FDE_D_FINAL")
    d_final_default = 1024 if d_final_env is None else int(d_final_env)
    d_final = args.fde_d_final if args.fde_d_final is not None else d_final_default

    seed = args.fde_seed if args.fde_seed is not None else _env_or_default("FDE_SEED", 42)

    cfg = FDEConfig(
        ksim=ksim,
        d_proj=d_proj,
        R_reps=r_reps,
        d_final=d_final,
        fill_empty_clusters=False, # Disabled for equivalence verification
        seed=seed,
        use_mixed_precision=False,
    )
    print(
        f"[PY] FDEConfig: ksim={cfg.ksim}, d_proj={cfg.d_proj}, "
        f"R_reps={cfg.R_reps}, d_final={cfg.d_final}, seed={cfg.seed}"
    )
    return cfg


def python_fde_embeddings(
    texts: List[str],
    bge_model: str,
    device: torch.device,
    fde_config: FDEConfig,
    max_length: int = 512,
    batch_size: int = 16,
    use_fp16: bool = False,
) -> np.ndarray:
    """
    Python 기준 파이프라인:
      text -> BGEM3FlagModel(colbert_vecs) -> FDEBuilder.encode_documents
    """
    encoder = BGEColBERTEncoder(
        model_name=bge_model,
        device=device,
        use_fp16=use_fp16,
        max_length=max_length,
        batch_size=batch_size,
    )

    print(f"[PY] Encoding {len(texts)} texts to ColBERT tokens...")
    token_lists = encoder.encode_texts(texts)  # list[Tensor(N_i, d)]

    if not token_lists:
        raise RuntimeError("No tokens produced from texts")

    d_embed = token_lists[0].shape[1]
    print(f"[PY] ColBERT dim = {d_embed}")
    builder = FDEBuilder(d=d_embed, config=fde_config, device=device)

    print("[PY] Running FDEBuilder.encode_documents ...")
    with torch.no_grad():
        fde = builder.encode_documents(token_lists)  # [B, d_fde]
        fde = fde.cpu().numpy().astype("float32")
    print(f"[PY] FDE embeddings shape: {fde.shape}")
    return fde


# -------------------------------------------------------------------
# 비교 / 리포트
# -------------------------------------------------------------------

def compare_embeddings(
    texts: List[str],
    emb_py: np.ndarray,
    emb_tei: np.ndarray,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    verbose: bool = True,
):
    if emb_py.shape != emb_tei.shape:
        print("[WARN] Shape mismatch:")
        print(f"  Python FDE: {emb_py.shape}")
        print(f"  TEI   FDE: {emb_tei.shape}")
    n, d = emb_py.shape[0], emb_py.shape[1]
    print(f"[COMPARE] num_texts={n}, dim_python={d}, dim_tei={emb_tei.shape[1]}")

    l2_list = []
    max_abs_list = []
    cos_list = []

    for i in range(n):
        v_py = emb_py[i]
        v_tei = emb_tei[i]

        diff = v_py - v_tei
        l2 = float(np.linalg.norm(diff))
        max_abs = float(np.max(np.abs(diff)))

        norm_py = float(np.linalg.norm(v_py) + 1e-12)
        norm_tei = float(np.linalg.norm(v_tei) + 1e-12)
        cos = float(np.dot(v_py, v_tei) / (norm_py * norm_tei))

        l2_list.append(l2)
        max_abs_list.append(max_abs)
        cos_list.append(cos)

        if verbose:
            print(f"\n[#{i}] text = {texts[i]!r}")
            print(f"  L2 diff      : {l2:.6e}")
            print(f"  max |diff|   : {max_abs:.6e}")
            print(f"  cos(python,tei): {cos:.6f}")

    l2_arr = np.array(l2_list)
    max_arr = np.array(max_abs_list)
    cos_arr = np.array(cos_list)

    print("\n========== SUMMARY ==========")
    print(f"  L2 diff   : mean={l2_arr.mean():.6e}, max={l2_arr.max():.6e}")
    print(f"  max|diff| : mean={max_arr.mean():.6e}, max={max_arr.max():.6e}")
    print(f"  cosine    : mean={cos_arr.mean():.6f}, min={cos_arr.min():.6f}")
    print("  allclose? :", bool(np.allclose(emb_py, emb_tei, atol=atol, rtol=rtol)))
    print(f"  (atol={atol}, rtol={rtol})")
    print("=============================")



# -------------------------------------------------------------------
# Save Params for TEI
# -------------------------------------------------------------------

def save_fde_params_for_tei(builder: FDEBuilder, filename: str = "fde_params.safetensors"):
    """
    Save FDEBuilder params (G, S, W) to safetensors format for TEI.
    Enforces S consistency (Rust uses single W for projection).
    """
    from safetensors.torch import save_file
    
    tensors = {}
    
    # 1. G: [R, d, k] -> [d, R*k] (Rust expects flattened ksim*r_reps)
    # Rust: g_shape = (hidden_size, config.ksim * config.r_reps)
    # Rust G is created as [hidden_size, ksim * r_reps]
    # Python G is [R, d, k]
    # We permute to [d, R, k] then reshape to [d, R*k]
    G_py = builder.params.G
    d = G_py.shape[1]
    G_rust = G_py.permute(1, 0, 2).reshape(d, -1).contiguous()
    tensors["g"] = G_rust
    
    # 2. W (Projector S): [R, d_proj, d] -> [d, d_proj]
    # Rust uses a single W [hidden_size, d_proj].
    # We must ensure Python uses the same S for all reps.
    # We take S[0] and replicate it for Python, and save S[0].T for Rust.
    if builder.params.S is not None:
        S_0 = builder.params.S[0] # [d_proj, d]
        # Enforce consistency in Python builder
        builder.params.S = S_0.unsqueeze(0).expand(builder.R, -1, -1).contiguous()
        
        # Save for Rust: [d, d_proj]
        W_rust = S_0.t().contiguous()
        tensors["w"] = W_rust
        
    # 3. P (Final Projection): [d_final, d_in] -> [d_in, d_final]
    # Rust P is [fde_dim, d_final]
    if builder.final_proj is not None:
        P_py = builder.final_proj.W # [d_final, d_in]
        P_rust = P_py.t().contiguous()
        tensors["p"] = P_rust
        
    print(f"[PY] Saving FDE params to {filename}...")
    save_file(tensors, filename)
    print(f"     G: {tensors['g'].shape}")
    if "w" in tensors: print(f"     W: {tensors['w'].shape}")
    if "p" in tensors: print(f"     P: {tensors['p'].shape}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Check BGE-M3 FDE equivalence: Python FDE vs TEI FDE"
    )

    # TEI
    parser.add_argument(
        "--tei-url",
        type=str,
        default="http://localhost:8080/v1/embeddings",
        help="TEI embeddings endpoint (e.g. http://localhost:8080/v1/embeddings)",
    )
    parser.add_argument(
        "--tei-model",
        type=str,
        default=None,
        help="Model name sent to TEI (default: omit, use server default)",
    )
    parser.add_argument(
        "--tei-api-key",
        type=str,
        default=None,
        help="Optional TEI API key (Authorization: Bearer ...)",
    )

    # Python side (BGE + FDE)
    parser.add_argument(
        "--bge-model",
        type=str,
        default="BAAI/bge-m3",
        help="BGEM3FlagModel HF name or local path",
    )
    parser.add_argument(
        "--bge-max-length",
        type=int,
        default=512,
        help="Max sequence length for BGEM3FlagModel",
    )
    parser.add_argument(
        "--bge-batch-size",
        type=int,
        default=16,
        help="Batch size for BGEM3FlagModel.encode",
    )
    parser.add_argument(
        "--bge-use-fp16",
        action="store_true",
        help="Use fp16 in BGEM3FlagModel",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for Python side (e.g. 'cuda', 'cuda:0', 'cpu'; default: auto)",
    )

    # FDE config (Python 쪽). 지정 안하면 환경변수/디폴트 사용.
    parser.add_argument("--fde-ksim", type=int, default=None)
    parser.add_argument("--fde-d-proj", type=int, default=None)
    parser.add_argument("--fde-r-reps", type=int, default=None)
    parser.add_argument("--fde-d-final", type=int, default=None)
    parser.add_argument("--fde-seed", type=int, default=None)

    # 텍스트 입력
    parser.add_argument(
        "--texts",
        type=str,
        nargs="*",
        default=None,
        help="직접 비교할 문장 목록. 공백으로 구분. 예: --texts 'hello' 'world'",
    )
    parser.add_argument(
        "--texts-file",
        type=str,
        default=None,
        help="텍스트 파일 경로. 한 줄 당 한 문장.",
    )

    # 비교 옵션
    parser.add_argument("--atol", type=float, default=1e-4, help="np.allclose atol")
    parser.add_argument("--rtol", type=float, default=1e-4, help="np.allclose rtol")
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="각 문장별 상세 diff 출력 생략",
    )
    
    parser.add_argument(
        "--save-params",
        type=str,
        default="fde_params.safetensors",
        help="Path to save FDE params for TEI",
    )

    args = parser.parse_args()

    # Device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Using Python device: {device}")

    # Texts
    texts: List[str] = []
    if args.texts is not None and len(args.texts) > 0:
        texts.extend(args.texts)
    if args.texts_file is not None:
        with open(args.texts_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)

    if not texts:
        # 기본 샘플
        texts = [
            "hello world",
            "the quick brown fox jumps over the lazy dog",
            "오늘 날씨 어때?",
            "BGE-M3 FDE equivalence test sentence",
        ]
        print("[INFO] No texts provided. Using default sample texts.")

    print(f"[INFO] #texts = {len(texts)}")
    for i, t in enumerate(texts):
        print(f"  [{i}] {t!r}")

    # FDE config (Python)
    fde_config = build_fde_config_from_env_and_args(args)

    # 1. Initialize Python FDE Builder
    # We need to initialize it HERE to save params before encoding
    # But `python_fde_embeddings` does it inside.
    # We will refactor slightly to do it outside or pass builder.
    
    # Let's manually init builder to save params
    # We need d_embed (hidden size). BGE-M3 is 1024.
    # But let's get it from model to be safe.
    encoder = BGEColBERTEncoder(
        model_name=args.bge_model,
        device=device,
        use_fp16=args.bge_use_fp16,
        max_length=args.bge_max_length,
        batch_size=args.bge_batch_size,
    )
    # Dummy encode to get dim
    dummy = encoder.encode_texts(["test"])
    d_embed = dummy[0].shape[1]
    print(f"[PY] Detected hidden size: {d_embed}")
    
    builder = FDEBuilder(d=d_embed, config=fde_config, device=device)
    
    # 2. Save params for TEI and enforce consistency
    save_fde_params_for_tei(builder, args.save_params)
    
    # 3. Python FDE Encoding (using the SAME builder)
    print(f"[PY] Encoding {len(texts)} texts to ColBERT tokens...")
    token_lists = encoder.encode_texts(texts)
    
    print("[PY] Running FDEBuilder.encode_documents ...")
    with torch.no_grad():
        fde = builder.encode_documents(token_lists)
        emb_py = fde.cpu().numpy().astype("float32")
    print(f"[PY] FDE embeddings shape: {emb_py.shape}")

    # 4. TEI FDE Encoding
    # Ensure TEI has reloaded the model with new params!
    # The user must restart TEI after saving params.
    print("\n[IMPORTANT] Please ensure TEI is restarted with the generated 'fde_params.safetensors' available to it!")
    input("Press Enter to continue after restarting TEI (or if it's already running with correct params)...")
    
    emb_tei = tei_embed(
        texts=texts,
        tei_url=args.tei_url,
        tei_model=args.tei_model,
        api_key=args.tei_api_key,
    )

    # Compare
    compare_embeddings(
        texts=texts,
        emb_py=emb_py,
        emb_tei=emb_tei,
        atol=args.atol,
        rtol=args.rtol,
        verbose=not args.no_verbose,
    )


if __name__ == "__main__":
    main()

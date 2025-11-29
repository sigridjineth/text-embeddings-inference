#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BGE-M3 ColBERT + FAST FDE 벤치마크 스크립트 (OOM 방지 개선 버전)

- BGE-M3 colbert_vecs 를 멀티벡터로 사용
- Baseline: ColBERT-style Chamfer(MaxSim)
- 여러 FDE 설정을 한 번에 비교:
    * FDE_k5_d16_R10_1024      (작은 FDE, final W 포함)
    * FDE_k5_d16_R10_raw       (같은 설정, final W 제거)
    * FDE_k5_d32_R10_2048      (조금 더 큰 FDE)
    * (옵션) FDE_DEBUG_k5_R6_no_proj_raw (더 큰 raw FDE, sanity check용)

사용 예시:
    python bge_m3_fast_fde_benchmark.py \
      --data-dir retrieval_benchmark_v2 \
      --corpus-file corpus.jsonl \
      --queries-file queries.jsonl \
      --qrels-file qrels/test.tsv \
      --bge-model BAAI/bge-m3 \
      --topk-list 1,3,5,10 \
      --include-debug-fde
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
from FlagEmbedding import BGEM3FlagModel

try:
    from fde_pytorch_opt import (
        FDEConfig,
        FDEBuilder,
        SamplerConfig,
        build_document_fde_batch,
        normalize_rows,
        chamfer_similarity,
    )
except ImportError:
    print("[WARNING] fde_pytorch_opt.py not found. Please ensure it is in the python path.")
    # Dummy imports to allow script to be written without immediate error
    FDEConfig = None
    FDEBuilder = None
    SamplerConfig = None
    build_document_fde_batch = None
    normalize_rows = None
    chamfer_similarity = None

# ---------------------------------------------------------
# IR Metrics
# ---------------------------------------------------------

def recall_at_k(ranked_ids: List[str], rel_ids: Set[str], k: int) -> float:
    if not rel_ids:
        return 0.0
    topk = ranked_ids[:k]
    hit = len(set(topk) & rel_ids)
    return hit / float(len(rel_ids))


def ndcg_at_k(ranked_ids: List[str], rel_scores: Dict[str, float], k: int) -> float:
    if not rel_scores:
        return 0.0

    def _dcg(scores: List[float]) -> float:
        s = 0.0
        for i, val in enumerate(scores):
            if val <= 0:
                continue
            s += (2.0 ** val - 1.0) / math.log2(i + 2.0)
        return s

    pred_scores = [rel_scores.get(doc_id, 0.0) for doc_id in ranked_ids[:k]]
    dcg_val = _dcg(pred_scores)

    ideal_scores = sorted(rel_scores.values(), reverse=True)[:k]
    idcg_val = _dcg(ideal_scores)
    if idcg_val == 0.0:
        return 0.0
    return dcg_val / idcg_val


# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------

def load_corpus(path: Path) -> Dict[str, Dict[str, str]]:
    corpus: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj["_id"]
            corpus[cid] = {
                "title": obj.get("title", "") or "",
                "text": obj.get("text", "") or "",
            }
    return corpus


def load_queries(path: Path) -> Dict[str, str]:
    queries: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj["_id"]
            queries[qid] = obj["text"]
    return queries


def load_qrels(path: Path) -> Dict[str, Dict[str, float]]:
    """
    qrels tsv:
      query_id \t corpus_id \t score
    """
    df = pd.read_csv(path, sep="\t")
    qrels: Dict[str, Dict[str, float]] = {}
    for row in df.itertuples(index=False):
        qid = row.query_id
        did = row.corpus_id
        score = float(row.score)
        if score <= 0:
            continue
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][did] = max(qrels[qid].get(did, 0.0), score)
    return qrels


# ---------------------------------------------------------
# BGE-M3 ColBERT Multi-vector Encoder
# ---------------------------------------------------------

class BGEColBERTEncoder:
    """
    BGE-M3의 colbert_vecs를 사용해
    텍스트 -> (m, d) 토큰 임베딩(Tensor) 리스트로 변환
    """

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        use_fp16: bool = True,
        max_length: int = 512,
        batch_size: int = 16,
    ):
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        print(f"[BGE] Loading BGEM3FlagModel: {model_name} (use_fp16={use_fp16})")
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

    def encode_texts(self, texts: List[str]) -> List[torch.Tensor]:
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
                if normalize_rows:
                    t = normalize_rows(t)
                all_tokens.append(t)
        return all_tokens


def build_bge_tokens(
    corpus: Dict[str, Dict[str, str]],
    queries: Dict[str, str],
    encoder: BGEColBERTEncoder,
) -> Tuple[List[str], List[str], List[torch.Tensor], Dict[str, torch.Tensor]]:
    doc_ids = list(corpus.keys())
    q_ids = list(queries.keys())

    doc_texts = [
        (corpus[did]["title"] + "\n" + corpus[did]["text"]).strip()
        for did in doc_ids
    ]
    print(f"[BGE] Encoding {len(doc_ids)} documents...")
    D_tokens_list = encoder.encode_texts(doc_texts)

    query_texts = [queries[qid] for qid in q_ids]
    print(f"[BGE] Encoding {len(q_ids)} queries...")
    Q_tokens_list = encoder.encode_texts(query_texts)
    Q_tokens_dict = {qid: Q_tokens_list[i] for i, qid in enumerate(q_ids)}

    return doc_ids, q_ids, D_tokens_list, Q_tokens_dict


# ---------------------------------------------------------
# Baseline: ColBERT Full Chamfer
# ---------------------------------------------------------

def evaluate_colbert_baseline(
    doc_ids: List[str],
    D_tokens_list: List[torch.Tensor],
    q_ids: List[str],
    Q_tokens_dict: Dict[str, torch.Tensor],
    qrels: Dict[str, Dict[str, float]],
    topk_list: List[int],
    device: torch.device,
) -> Dict[int, Dict[str, float]]:
    print("\n[BASELINE] Evaluating ColBERT Chamfer (full multi-vector)...")
    metrics = {k: {"recall": [], "ndcg": []} for k in topk_list}

    n_eval = 0
    t0 = time.perf_counter()

    for qid in q_ids:
        if qid not in qrels:
            continue
        rel_dict = qrels[qid]
        rel_docs = set(rel_dict.keys())
        if not rel_docs:
            continue

        q_tokens = Q_tokens_dict[qid].to(device)

        scores = []
        for d_tokens in D_tokens_list:
            if chamfer_similarity:
                s = chamfer_similarity(q_tokens, d_tokens.to(device))
                scores.append(s)
        scores = np.array(scores)
        ranked_idx = np.argsort(-scores)
        ranked_doc_ids = [doc_ids[i] for i in ranked_idx]

        for K in topk_list:
            r = recall_at_k(ranked_doc_ids, rel_docs, K)
            n = ndcg_at_k(ranked_doc_ids, rel_dict, K)
            metrics[K]["recall"].append(r)
            metrics[K]["ndcg"].append(n)

        n_eval += 1

    t1 = time.perf_counter()
    print(f"[BASELINE] ColBERT evaluated {n_eval} queries in {t1 - t0:.2f}s "
          f"({(t1 - t0) / max(1, n_eval):.4f}s / query)")
    return metrics


# ---------------------------------------------------------
# FDE Variant Evaluation
# ---------------------------------------------------------

def evaluate_fde_variant(
    name: str,
    fde_cfg: 'FDEConfig',
    doc_ids: List[str],
    D_tokens_list: List[torch.Tensor],
    q_ids: List[str],
    Q_tokens_dict: Dict[str, torch.Tensor],
    qrels: Dict[str, Dict[str, float]],
    topk_list: List[int],
    device: torch.device,
    batch_size_docs: int = 64,
) -> Tuple[Dict[int, Dict[str, float]], float, float]:
    print(f"\n[FDE:{name}] Config: "
          f"ksim={fde_cfg.ksim}, d_proj={fde_cfg.d_proj}, "
          f"R={fde_cfg.R_reps}, d_final={fde_cfg.d_final}, "
          f"fill_empty={fde_cfg.fill_empty_clusters}, "
          f"mixed={fde_cfg.use_mixed_precision}")

    d_embed = D_tokens_list[0].shape[1]
    if FDEBuilder:
        builder = FDEBuilder(d=d_embed, config=fde_cfg, device=device)
    else:
        print("FDEBuilder not found")
        return {}, 0.0, 0.0

    if SamplerConfig:
        refine_sampler = SamplerConfig(
            token_sample_ratio=1.0,
            per_bucket_cap=0,
            use_pps=False,
            r_sub=0,
            seed=123,
        )
    else:
        refine_sampler = None

    # 1) 문서 FDE 인덱스 빌드
    t0 = time.perf_counter()
    if build_document_fde_batch:
        doc_fdes_list = build_document_fde_batch(
            D_tokens_list,
            builder,
            sampler=refine_sampler,
            batch_size_docs=batch_size_docs,
            r_sub_override=0,
        )
        doc_fdes = torch.stack(doc_fdes_list, dim=0).to(device)
    else:
        doc_fdes = torch.tensor([])
    
    t1 = time.perf_counter()
    build_time = t1 - t0
    print(f"[FDE:{name}] Doc FDE built: shape={tuple(doc_fdes.shape)}, time={build_time:.2f}s")

    metrics = {k: {"recall": [], "ndcg": []} for k in topk_list}

    # 2) 쿼리 평가
    t2 = time.perf_counter()
    n_eval = 0
    for qid in q_ids:
        if qid not in qrels:
            continue
        rel_dict = qrels[qid]
        rel_docs = set(rel_dict.keys())
        if not rel_docs:
            continue

        q_tokens = Q_tokens_dict[qid].to(device)
        q_fde = builder.build_query_fde(q_tokens)

        scores = (doc_fdes @ q_fde).detach().cpu().numpy()
        ranked_idx = np.argsort(-scores)
        ranked_doc_ids = [doc_ids[i] for i in ranked_idx]

        for K in topk_list:
            r = recall_at_k(ranked_doc_ids, rel_docs, K)
            n = ndcg_at_k(ranked_doc_ids, rel_dict, K)
            metrics[K]["recall"].append(r)
            metrics[K]["ndcg"].append(n)

        n_eval += 1
    t3 = time.perf_counter()
    eval_time = t3 - t2
    print(f"[FDE:{name}] Evaluated {n_eval} queries in {eval_time:.2f}s "
          f"({eval_time / max(1, n_eval):.4f}s / query)")

    # 3) GPU 메모리 정리 (다음 variant를 위해)
    if device.type == "cuda":
        del doc_fdes
        torch.cuda.empty_cache()

    return metrics, build_time, eval_time


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BGE-M3 ColBERT + FAST FDE Benchmark (OOM-safe)")

    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--corpus-file", type=str, default="corpus.jsonl")
    parser.add_argument("--queries-file", type=str, default="queries.jsonl")
    parser.add_argument("--qrels-file", type=str, default="qrels/test.tsv")

    parser.add_argument("--bge-model", type=str, default="BAAI/bge-m3",
                        help="BGE-M3 HuggingFace model name")
    parser.add_argument("--bge-max-length", type=int, default=512)
    parser.add_argument("--bge-batch-size", type=int, default=16)
    parser.add_argument("--bge-use-fp16", action="store_true")

    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu (default: auto)")
    parser.add_argument("--topk-list", type=str, default="1,3,5,10",
                        help="예: '1,3,5,10'")

    parser.add_argument("--skip-colbert-baseline", action="store_true",
                        help="ColBERT Chamfer baseline 생략")
    parser.add_argument("--fde-batch-size-docs", type=int, default=64)

    parser.add_argument("--include-debug-fde", action="store_true",
                        help="더 큰 raw FDE(DEBUG)를 추가로 평가 (메모리/시간 더 사용)")

    args = parser.parse_args()

    # Device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    # TopK list
    topk_list = sorted({int(x) for x in args.topk_list.split(",") if x.strip()})
    print(f"[INFO] Evaluate K values: {topk_list}")

    # Load data
    data_dir = Path(args.data_dir)
    corpus_path = data_dir / args.corpus_file
    queries_path = data_dir / args.queries_file
    qrels_path = data_dir / args.qrels_file

    print(f"[INFO] Loading data from {data_dir} ...")
    corpus = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)
    print(f"[INFO] #docs={len(corpus)}, #queries={len(queries)}, "
          f"#qrels_queries={len(qrels)}")

    # BGE-M3 ColBERT encoder
    encoder = BGEColBERTEncoder(
        model_name=args.bge_model,
        device=device,
        use_fp16=args.bge_use_fp16,
        max_length=args.bge_max_length,
        batch_size=args.bge_batch_size,
    )

    # Multi-vector tokens
    doc_ids, q_ids, D_tokens_list, Q_tokens_dict = build_bge_tokens(
        corpus, queries, encoder
    )
    print(f"[INFO] Got BGE-M3 ColBERT tokens: "
          f"{len(D_tokens_list)} docs, {len(Q_tokens_dict)} queries")
    print(f"[INFO] ColBERT token dimension (d) = {D_tokens_list[0].shape[1]}")

    # BGE 모델은 더 이상 안 쓰이므로 GPU 메모리에서 제거
    if hasattr(encoder, "model"):
        del encoder.model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print("[INFO] Freed BGE-M3 model from GPU to save VRAM.")

    results = {}

    # --------------------------
    # Baseline: ColBERT Chamfer
    # --------------------------
    if not args.skip_colbert_baseline:
        baseline_metrics = evaluate_colbert_baseline(
            doc_ids,
            D_tokens_list,
            q_ids,
            Q_tokens_dict,
            qrels,
            topk_list,
            device,
        )
        results["ColBERT"] = {
            "metrics": baseline_metrics,
            "build_time": 0.0,
            "eval_time": 0.0,
        }

    # --------------------------
    # FDE variants 설정
    # --------------------------
    fde_variants: List[Dict] = []

    if FDEConfig:
        # 1) 작은 FDE (논문보다 aggressive한 압축)
        fde_variants.append({
            "name": "FDE_k5_d16_R10_1024",
            "cfg": FDEConfig(
                ksim=5,
                d_proj=16,
                R_reps=10,
                d_final=1024,
                fill_empty_clusters=True,
                seed=123,
                use_mixed_precision=False,
            ),
        })

        # 2) 1)과 동일하지만 final projection(W) 제거 → raw FDE
        fde_variants.append({
            "name": "FDE_k5_d16_R10_raw",
            "cfg": FDEConfig(
                ksim=5,
                d_proj=16,
                R_reps=10,
                d_final=None,
                fill_empty_clusters=True,
                seed=123,
                use_mixed_precision=False,
            ),
        })

        # 3) 조금 더 큰 FDE: d_proj=32, d_final=2048
        fde_variants.append({
            "name": "FDE_k5_d32_R10_2048",
            "cfg": FDEConfig(
                ksim=5,
                d_proj=32,
                R_reps=10,
                d_final=2048,
                fill_empty_clusters=True,
                seed=123,
                use_mixed_precision=False,
            ),
        })

        # 4) DEBUG FDE: muvera 스타일에 더 가까운 "큰" raw FDE
        #    - ksim=5, d_proj=0 → per-bucket에서 d 그대로 사용
        #    - R=6 → FDE dim = 6 * 32 * d ≈ 196k (BGE d=1024 기준)
        #    - d_final=None → 추가 사영 없음
        if args.include_debug_fde:
            fde_variants.append({
                "name": "FDE_DEBUG_k5_R6_no_proj_raw",
                "cfg": FDEConfig(
                    ksim=5,
                    d_proj=0,
                    R_reps=6,
                    d_final=None,
                    fill_empty_clusters=True,
                    seed=42,
                    use_mixed_precision=False,
                ),
            })
            print("[INFO] DEBUG FDE enabled: ksim=5, R=6, d_proj=0, d_final=None "
                  "(larger raw FDE, more VRAM and time).")

    # --------------------------
    # 각 FDE 버전 평가
    # --------------------------
    for variant in fde_variants:
        name = variant["name"]
        cfg = variant["cfg"]
        metrics, build_t, eval_t = evaluate_fde_variant(
            name,
            cfg,
            doc_ids,
            D_tokens_list,
            q_ids,
            Q_tokens_dict,
            qrels,
            topk_list,
            device,
            batch_size_docs=args.fde_batch_size_docs,
        )
        results[name] = {
            "metrics": metrics,
            "build_time": build_t,
            "eval_time": eval_t,
        }

    # --------------------------
    # 요약 출력
    # --------------------------
    print("\n================ BENCHMARK SUMMARY ================")
    for name, info in results.items():
        print(f"\n[{name}]")
        mt = info["metrics"]
        for K in topk_list:
            if K not in mt:
                continue
            r_list = mt[K]["recall"]
            n_list = mt[K]["ndcg"]
            if not r_list:
                continue
            avg_r = sum(r_list) / len(r_list)
            avg_n = sum(n_list) / len(n_list)
            print(f"  K={K:>3} | Recall@{K}: {avg_r:.4f} | nDCG@{K}: {avg_n:.4f}")
        if name != "ColBERT":
            print(f"  build_time: {info['build_time']:.2f}s, "
                  f"eval_time: {info['eval_time']:.2f}s")
    print("===================================================")


if __name__ == "__main__":
    main()

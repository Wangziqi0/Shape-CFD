#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
complementarity_analysis.py — Reviewer 2 (ii) 互补性悖论深度分析

核心问题:
    Reviewer 2 指出 Kendall tau=0.378 被作者声称为 high complementarity,
    但 weighted fusion 没拿到统计显著提升. 这是悖论吗?

回答:
    Kendall tau 度量 RANK correlation, 不预测 SCORE-UPDATE direction.
    两个 method 可能 rank 上不同 (低 tau), 但相对 cosine baseline 的 score-update
    方向高度相关 (高 cosine similarity of update vectors), 此时 fusion 无法叠加.

    真正的互补性应当度量 score-update orthogonality:
        delta_token = score_token - score_cosine
        delta_lap   = score_lap   - score_cosine
        complementarity = 1 - |cos(delta_token, delta_lap)|

    cos ~ 1 (parallel updates): fusion 无 gain (两 method 推同方向)
    cos ~ 0 (orthogonal):       fusion 应有 gain (两 method 探不同维度)
    cos ~ -1 (anti-parallel):   fusion 互相抵消

输入:
    /home/amd/HEZIMENG/Shape-CFD/benchmark/data/results/fusion_ablation_<DATASET>_perquery.jsonl

输出:
    /home/amd/HEZIMENG/Shape-CFD/benchmark/data/results/complementarity_analysis_<DATASET>.json

本实现 NDCG-level (cheap, no raw score needed).
若需 raw-score-level 分析, 后续扩展 Node script 落 raw scores 再处理.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_perquery_jsonl(fp: Path) -> List[Dict]:
    rows: List[Dict] = []
    with fp.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def analyze_dataset(perquery_file: Path) -> Dict:
    rows = load_perquery_jsonl(perquery_file)
    print(f"loaded {len(rows)} per-query rows from {perquery_file}")

    qids: List[str] = []
    cos_l: List[float] = []
    tok_l: List[float] = []
    lap5_l: List[float] = []
    fus_lam: Dict[str, List[float]] = {f"{x:.1f}": [] for x in np.arange(0.0, 1.01, 0.1)}
    tau_per: List[float] = []
    rho_per: List[float] = []

    for r in rows:
        n = r.get("ndcg", {})
        if "cosine" not in n or "token_2stage" not in n or "lap_T5" not in n:
            continue
        qids.append(r["qid"])
        cos_l.append(n["cosine"])
        tok_l.append(n["token_2stage"])
        lap5_l.append(n["lap_T5"])
        for k in fus_lam:
            kk = f"fusion_T5_lam{k}"
            fus_lam[k].append(n.get(kk, np.nan))
        if "kendall_tau" in r and r["kendall_tau"] is not None:
            tau_per.append(r["kendall_tau"])
        if "spearman_rho" in r and r["spearman_rho"] is not None:
            rho_per.append(r["spearman_rho"])

    cos = np.asarray(cos_l, dtype=np.float64)
    tok = np.asarray(tok_l, dtype=np.float64)
    lap5 = np.asarray(lap5_l, dtype=np.float64)
    nq = len(cos)
    print(f"valid query rows: {nq}")

    delta_tok = tok - cos
    delta_lap = lap5 - cos

    update_orthogonality = {
        "delta_token_mean": float(delta_tok.mean()),
        "delta_token_std": float(delta_tok.std()),
        "delta_lap_mean": float(delta_lap.mean()),
        "delta_lap_std": float(delta_lap.std()),
        "cos_delta_full": cosine_sim(delta_tok, delta_lap),
        "pearson_delta": float(np.corrcoef(delta_tok, delta_lap)[0, 1]) if nq > 1 else float("nan"),
    }

    quad = {
        "both_positive": int(((delta_tok > 0) & (delta_lap > 0)).sum()),
        "both_negative": int(((delta_tok < 0) & (delta_lap < 0)).sum()),
        "tok_pos_lap_neg": int(((delta_tok > 0) & (delta_lap < 0)).sum()),
        "tok_neg_lap_pos": int(((delta_tok < 0) & (delta_lap > 0)).sum()),
        "tok_zero": int((delta_tok == 0).sum()),
        "lap_zero": int((delta_lap == 0).sum()),
    }

    tau_arr = np.asarray(tau_per, dtype=np.float64) if tau_per else np.asarray([])
    rho_arr = np.asarray(rho_per, dtype=np.float64) if rho_per else np.asarray([])
    tau_summary = {
        "n": int(len(tau_arr)),
        "mean": float(tau_arr.mean()) if len(tau_arr) else None,
        "std": float(tau_arr.std()) if len(tau_arr) else None,
        "median": float(np.median(tau_arr)) if len(tau_arr) else None,
    }
    rho_summary = {
        "n": int(len(rho_arr)),
        "mean": float(rho_arr.mean()) if len(rho_arr) else None,
        "std": float(rho_arr.std()) if len(rho_arr) else None,
    }

    fusion_summary: Dict[str, Dict] = {}
    for k, vs in fus_lam.items():
        arr = np.asarray([v for v in vs if not (isinstance(v, float) and math.isnan(v))], dtype=np.float64)
        if len(arr) > 0:
            fusion_summary[k] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "n": int(len(arr)),
                "delta_vs_token": float(arr.mean() - tok.mean()),
                "delta_vs_lap": float(arr.mean() - lap5.mean()),
            }

    best_lam = max(fusion_summary, key=lambda k: fusion_summary[k]["mean"]) if fusion_summary else None
    best_individual = "token" if tok.mean() > lap5.mean() else "lap"
    best_individual_arr = tok if best_individual == "token" else lap5
    boot_test = None
    sign_analysis = None
    if best_lam is not None:
        best_fusion_arr = np.asarray(fus_lam[best_lam], dtype=np.float64)
        valid = ~np.isnan(best_fusion_arr)
        bf = best_fusion_arr[valid]
        bi = best_individual_arr[valid]
        diff = bf - bi
        rng = np.random.default_rng(42)
        n_boot = 10000
        boot = np.empty(n_boot, dtype=np.float64)
        for i in range(n_boot):
            idx = rng.integers(0, len(diff), size=len(diff))
            boot[i] = diff[idx].mean()
        mean_diff = float(diff.mean())
        if mean_diff >= 0:
            p = float(2 * (boot <= 0).mean())
        else:
            p = float(2 * (boot >= 0).mean())
        p = min(p, 1.0)
        boot_test = {
            "best_fusion_lambda": best_lam,
            "best_fusion_ndcg": float(bf.mean()),
            "best_individual_method": best_individual,
            "best_individual_ndcg": float(bi.mean()),
            "mean_diff": mean_diff,
            "ci_low_95": float(np.percentile(boot, 2.5)),
            "ci_high_95": float(np.percentile(boot, 97.5)),
            "p_value_two_sided": p,
            "n_queries": int(len(diff)),
        }

        # sign analysis
        same_sign_mask = ((delta_tok > 0) & (delta_lap > 0)) | ((delta_tok < 0) & (delta_lap < 0))
        diff_sign_mask = ((delta_tok > 0) & (delta_lap < 0)) | ((delta_tok < 0) & (delta_lap > 0))
        gain_vs_max = best_fusion_arr - np.maximum(tok, lap5)
        sign_analysis = {
            "n_same_sign": int(same_sign_mask.sum()),
            "n_diff_sign": int(diff_sign_mask.sum()),
            "fusion_gain_same_sign_mean": float(gain_vs_max[same_sign_mask & valid].mean()) if (same_sign_mask & valid).any() else None,
            "fusion_gain_diff_sign_mean": float(gain_vs_max[diff_sign_mask & valid].mean()) if (diff_sign_mask & valid).any() else None,
        }

    return {
        "n_queries": nq,
        "ndcg_means": {
            "cosine": float(cos.mean()),
            "token": float(tok.mean()),
            "lap_T5": float(lap5.mean()),
        },
        "update_orthogonality": update_orthogonality,
        "quadrant_analysis": quad,
        "kendall_tau_per_query": tau_summary,
        "spearman_rho_per_query": rho_summary,
        "fusion_lambda_sweep": fusion_summary,
        "bootstrap_best_fusion_vs_best_individual": boot_test,
        "sign_analysis": sign_analysis,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--input-dir", default="/home/amd/HEZIMENG/Shape-CFD/benchmark/data/results")
    p.add_argument("--output-dir", default="/home/amd/HEZIMENG/Shape-CFD/benchmark/data/results")
    args = p.parse_args()

    perquery_file = Path(args.input_dir) / f"fusion_ablation_{args.dataset}_perquery.jsonl"
    if not perquery_file.exists():
        raise SystemExit(f"missing: {perquery_file}")

    summary = analyze_dataset(perquery_file)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"complementarity_analysis_{args.dataset}.json"
    with out_file.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_file}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

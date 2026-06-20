#!/usr/bin/env python3
"""
评估模块 — GRAG记忆系统指标计算

提供跨主题检索、图结构、PPR收敛等评估指标的
精确计算函数，确保实验结果可复现和可量化。
"""
import json
import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skills.memory_skill import MemoryGraph


def evaluate_cross_topic_retrieval(mem: MemoryGraph,
                                   queries: list[tuple[str, list[str]]],
                                   top_k: int = 5) -> dict:
    """
    跨主题检索评估

    对每个查询执行GRAG检索，统计：
    - 多主题覆盖率: 检索结果涉及的主题数 > 1 的比例
    - 期望主题覆盖率: 期望主题出现在检索结果中的比例
    - 主题熵: 检索结果主题分布的多样性

    Args:
        mem: 初始化好的MemoryGraph实例
        queries: [(query, [expected_topics]), ...]
        top_k: 检索top-k结果

    Returns:
        dict: 评估指标
    """
    results = []
    for query, expected_topics in queries:
        grag_results = mem.search(query, top_k=top_k)
        topics_found = list(set(r["topic"] for r in grag_results))
        covered = [t for t in expected_topics if t in topics_found]
        coverage = len(covered) / len(expected_topics)

        results.append({
            "query": query,
            "expected_topics": expected_topics,
            "topics_found": topics_found,
            "covered_topics": covered,
            "coverage": round(coverage, 4),
            "is_multi_topic": len(topics_found) > 1,
        })

    multi_topic_rate = sum(1 for r in results if r["is_multi_topic"]) / len(results)
    avg_coverage = np.mean([r["coverage"] for r in results])

    return {
        "num_queries": len(queries),
        "multi_topic_rate": round(multi_topic_rate, 4),
        "avg_coverage": round(avg_coverage, 4),
        "detail": results,
    }


def evaluate_ppr_boost(mem: MemoryGraph,
                       queries: list[str],
                       top_k: int = 5) -> dict:
    """
    PPR增益评估

    对比纯向量检索和GRAG检索的分数差异，量化PPR传播的增益。

    Args:
        mem: MemoryGraph实例
        queries: 查询列表
        top_k: 检索数量

    Returns:
        dict: PPR增益统计
    """
    emb_arr = np.array(mem.embeddings)
    boosts = []

    for query in queries:
        qv = mem.embedder.encode_one(query)
        vec_scores = np.dot(emb_arr, qv)

        # 纯向量top-k分数
        vec_top = np.sort(vec_scores)[::-1][:top_k]

        # GRAG top-k分数
        grag_results = mem.search(query, top_k=top_k)
        grag_top = np.array([r["score"] for r in grag_results])

        avg_boost = np.mean(grag_top - vec_top[:len(grag_top)])
        boosts.append(round(float(avg_boost), 4))

    return {
        "avg_ppr_boost": round(float(np.mean(boosts)), 4),
        "max_ppr_boost": max(boosts),
        "per_query": boosts,
    }


def evaluate_graph_structure(mem: MemoryGraph) -> dict:
    """
    图结构评估

    计算图的各种结构属性。

    Returns:
        dict: 图结构指标
    """
    if len(mem.nodes) == 0:
        return {}

    emb_arr = np.array(mem.embeddings)
    n = len(mem.nodes)
    e = sum(len(v) for v in mem.adj_list.values()) // 2
    degrees = [len(mem.adj_list[i]) for i in range(n)]

    # 平均聚类系数（简化为节点间的三角形计数）
    triangles = 0
    for i in range(n):
        neighbors = mem.adj_list[i]
        for j in neighbors:
            if j > i:
                common = neighbors & mem.adj_list[j]
                triangles += len(common)
    # 每个三角形被计数3次
    clustering_coeff = triangles / (n * (n - 1) / 2) if n > 1 else 0

    return {
        "num_nodes": n,
        "num_edges": e,
        "graph_density": round(e / (n * (n - 1) / 2) * 100, 2) if n > 1 else 0,
        "avg_degree": round(np.mean(degrees), 2),
        "max_degree": max(degrees),
        "min_degree": min(degrees),
        "clustering_coefficient": round(clustering_coeff, 4),
    }


def evaluate_ppr_convergence(mem: MemoryGraph, query: str) -> dict:
    """
    PPR收敛性评估

    测量PPR迭代收敛的详细过程。

    Returns:
        dict: 收敛过程数据
    """
    qv = mem.embedder.encode_one(query)
    emb_arr = np.array(mem.embeddings)
    vec_scores = np.dot(emb_arr, qv)

    pv = vec_scores.copy()
    s = pv.sum()
    if s > 0:
        pv = pv / s

    # 记录每一轮的分数变化
    scores = pv.copy()
    convergence = []
    for t in range(mem.ppr_max_iter):
        new_scores = np.zeros(len(mem.nodes))
        for i in range(len(mem.nodes)):
            if mem.adj_list[i]:
                ns = sum(scores[j] / max(len(mem.adj_list[j]), 1)
                         for j in mem.adj_list[i])
                new_scores[i] = (1 - mem.ppr_alpha) * ns
            new_scores[i] += mem.ppr_alpha * pv[i]
        diff = float(np.sum(np.abs(new_scores - scores)))
        convergence.append({"iteration": t + 1, "l1_change": diff})
        if diff < mem.ppr_convergence:
            break
        scores = new_scores

    return {
        "total_iterations": len(convergence),
        "final_l1_change": convergence[-1]["l1_change"] if convergence else 0,
        "convergence_history": convergence,
    }


def compute_term_recall(query: str, context: str, expected_terms: list[str]) -> float:
    """
    Term Recall: 期望术语在检索结果中出现的比例

    Args:
        query: 原始查询
        context: 检索到的上下文文本
        expected_terms: 期望出现的术语列表

    Returns:
        float: Term Recall (0~1)
    """
    if not expected_terms:
        return 1.0
    context_lower = context.lower()
    matched = sum(1 for t in expected_terms if t.lower() in context_lower)
    return matched / len(expected_terms)


if __name__ == "__main__":
    mem = MemoryGraph({"storage_path": "results/experiment_memories.json"})
    if len(mem.nodes) == 0:
        print("No memories found. Run experiments first.")
        sys.exit(1)

    report = {
        "graph_structure": evaluate_graph_structure(mem),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

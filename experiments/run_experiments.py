#!/usr/bin/env python3
"""
实验运行器 — 完整的 GRAG 记忆系统实验流程

实验方法：
  1. 构建5主题17轮对话记忆图
  2. 执行纯向量 vs GRAG 检索对比
  3. 评估跨主题检索能力
  4. 分析图结构属性
  5. 测试边界情况

评估指标：
  - Topic Match Count: top-5中相关主题命中数
  - Cross-Topic Coverage: 多主题检索覆盖率
  - PPR Convergence: 收敛迭代次数
  - Graph Density: 图密度和平均度
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from skills.memory_skill import MemoryGraph, EmbeddingModel


# ─── 实验数据 ────────────────────────────────────────────

TEST_CONVERSATIONS = [
    # (topic, user_msg, assistant_msg)
    ("rag", "什么是RAG？", "RAG是检索增强生成，结合检索和文本生成的技术。"),
    ("rag", "RAG有哪些组件？", "RAG包含检索器、知识库和生成器三个核心组件。"),
    ("rag", "FAISS是什么？", "FAISS是Meta开发的向量相似度搜索库。"),
    ("rag", "embedding模型的作用？", "embedding模型将文本转为向量用于语义相似度计算。"),
    ("rag", "如何评估RAG系统？", "RAG评估包括Recall@K和ROUGE/BLEU。"),

    ("python", "列表和元组区别？", "列表可变，元组不可变。列表用[]元组用()。"),
    ("python", "什么是装饰器？", "装饰器是扩展函数功能的高阶函数。"),
    ("python", "Python的GIL？", "GIL限制同一时刻只有一个线程执行字节码。"),

    ("deep-learning", "什么是反向传播？", "反向传播通过链式法则计算梯度更新参数。"),
    ("deep-learning", "Transformer自注意力？", "自注意力计算序列中位置间的关联权重。"),
    ("deep-learning", "什么是过拟合？", "过拟合是训练集好测试集差的现象。"),

    ("database", "数据库索引？", "数据库索引是加速数据检索的数据结构。"),
    ("database", "SQL和NoSQL区别？", "SQL关系型，NoSQL非关系型，适用不同场景。"),

    ("nlp", "什么是词嵌入？", "词嵌入将词语映射到低维稠密向量。"),
    ("nlp", "BERT怎么训练的？", "BERT使用掩码语言模型和下一句预测预训练。"),
    ("nlp", "什么是Seq2Seq？", "Seq2Seq将输入序列映射到输出序列。"),

    ("rag", "GRAG相比RAG的优势？", "GRAG通过图传播增强检索，支持多跳推理。"),
]

CROSS_TOPIC_QUERIES = [
    ("如何在Python中实现深度学习模型？", ["python", "deep-learning"]),
    ("向量数据库如何加速RAG检索？", ["database", "rag"]),
    ("NLP中的词嵌入与深度学习的关系", ["nlp", "deep-learning"]),
    ("数据库索引和FAISS索引的异同", ["database", "rag"]),
    ("如何评估NLP模型的性能？", ["nlp", "rag"]),
]

VECTOR_VS_GRAG_QUERIES = [
    ("RAG检索增强技术", "rag"),
    ("Python并发编程", "python"),
    ("深度学习模型训练", "deep-learning"),
    ("数据库性能优化", "database"),
    ("NLP嵌入表示学习", "nlp"),
]


# ─── 实验1: 构建记忆图 ──────────────────────────────────

def experiment_build_graph(mem: MemoryGraph) -> dict:
    """构建记忆图并返回统计"""
    print("\n" + "=" * 70)
    print("  [实验1] 构建记忆图")
    print("=" * 70)

    for topic, usr, asst in TEST_CONVERSATIONS:
        mem.store_conversation(usr, asst, topic=topic)

    stats = mem.stats()
    print(f"  节点: {stats['total_memories']}")
    print(f"  边: {stats['edges']}")
    print(f"  密度: {stats['density']}%")
    print(f"  平均度: {stats['avg_degree']}")
    print(f"  主题: {stats['topics']}")
    return stats


# ─── 实验2: 向量 vs GRAG 检索对比 ──────────────────────

def experiment_vector_vs_grag(mem: MemoryGraph) -> list[dict]:
    """对比纯向量检索和GRAG检索的主题匹配率"""
    print("\n" + "=" * 70)
    print("  [实验2] 纯向量 vs GRAG 检索对比")
    print("=" * 70)

    results = []
    for query, expected_topic in VECTOR_VS_GRAG_QUERIES:
        emb_arr = np.array(mem.embeddings)
        qv = mem.embedder.encode_one(query)
        vec_scores = np.dot(emb_arr, qv)
        top_vec = np.argsort(vec_scores)[::-1][:5]
        grag_results = mem.search(query, top_k=5)

        vec_match = sum(1 for i in top_vec if mem.nodes[i].topic == expected_topic)
        grag_match = sum(1 for r in grag_results if r["topic"] == expected_topic)

        results.append({
            "query": query,
            "expected_topic": expected_topic,
            "vector_match": vec_match,
            "grag_match": grag_match,
            "improvement": grag_match - vec_match,
        })

        print(f"\n  Query: {query}")
        print(f"    Vector Match: {vec_match}/5")
        print(f"    GRAG Match:   {grag_match}/5")
        print(f"    Improvement:  {'+' if grag_match > vec_match else ''}{grag_match - vec_match}")

    avg_vec = np.mean([r["vector_match"] for r in results])
    avg_grag = np.mean([r["grag_match"] for r in results])
    print(f"\n  >>> Average: Vector={avg_vec:.1f}/5, GRAG={avg_grag:.1f}/5")
    return results


# ─── 实验3: 跨主题检索 ──────────────────────────────────

def experiment_cross_topic(mem: MemoryGraph) -> list[dict]:
    """评估GRAG的跨主题检索能力"""
    print("\n" + "=" * 70)
    print("  [实验3] 跨主题检索评估")
    print("=" * 70)

    results = []
    for query, expected_topics in CROSS_TOPIC_QUERIES:
        grag_results = mem.search(query, top_k=5)
        topics_found = list(set(r["topic"] for r in grag_results))
        covered = sum(1 for t in expected_topics if t in topics_found)
        multi_topic = len(topics_found) > 1

        results.append({
            "query": query,
            "expected_topics": expected_topics,
            "topics_found": topics_found,
            "covered": covered,
            "multi_topic": multi_topic,
        })

        print(f"\n  Query: {query}")
        print(f"    Expected: {expected_topics}")
        print(f"    Found:    {topics_found}")
        print(f"    Covered:  {covered}/{len(expected_topics)} "
              f"{'✓' if multi_topic else '✗'}")

    success = sum(1 for r in results if r["multi_topic"])
    print(f"\n  >>> Multi-topic coverage: {success}/{len(results)} ({success/len(results)*100:.0f}%)")
    return results


# ─── 实验4: 图结构分析 ─────────────────────────────────

def experiment_graph_analysis(mem: MemoryGraph) -> dict:
    """分析图结构属性"""
    print("\n" + "=" * 70)
    print("  [实验4] 图结构分析")
    print("=" * 70)

    stats = mem.stats()
    n = stats["total_memories"]
    e = stats["edges"]

    # 度分布
    degrees = [len(mem.adj_list[i]) for i in range(n)] if n > 0 else []
    max_deg = max(degrees) if degrees else 0
    min_deg = min(degrees) if degrees else 0

    # 枢纽节点（度 >= max_deg * 0.6）
    hub_threshold = max_deg * 0.6
    hubs = [(i, degrees[i], mem.nodes[i].topic, mem.nodes[i].content[:40])
            for i in range(n) if degrees[i] >= hub_threshold]

    print(f"  总节点: {n}")
    print(f"  总边数: {e}")
    print(f"  密度: {stats['density']}%")
    print(f"  平均度: {stats['avg_degree']}")
    print(f"  度范围: [{min_deg}, {max_deg}]")

    if hubs:
        print(f"\n  枢纽节点 (度>={hub_threshold:.0f}):")
        for nid, deg, topic, content in sorted(hubs, key=lambda x: x[1], reverse=True)[:5]:
            print(f"    #{nid} [{topic}] deg={deg} | {content}...")

    return {
        "graph_density": stats["density"],
        "avg_degree": stats["avg_degree"],
        "max_degree": max_deg,
        "min_degree": min_deg,
        "hub_count": len(hubs),
    }


# ─── 实验5: PPR收敛性 ──────────────────────────────────

def experiment_ppr_convergence(mem: MemoryGraph) -> dict:
    """测试PPR传播收敛速度"""
    print("\n" + "=" * 70)
    print("  [实验5] PPR收敛性测试")
    print("=" * 70)

    iters_list = []
    for query, _ in VECTOR_VS_GRAG_QUERIES:
        qv = mem.embedder.encode_one(query)
        emb_arr = np.array(mem.embeddings)
        vec_s = np.dot(emb_arr, qv)
        _, n_iters = mem._ppr(vec_s)
        iters_list.append(n_iters)
        print(f"  Query '{query[:15]}...': {n_iters} iterations")

    import time
    start = time.time()
    _ = mem.search("RAG技术", top_k=5)
    latency = (time.time() - start) * 1000

    result = {
        "avg_iters": round(np.mean(iters_list), 1),
        "min_iters": min(iters_list),
        "max_iters": max(iters_list),
        "latency_ms": round(latency, 1),
    }
    print(f"\n  >>> Avg iterations: {result['avg_iters']}")
    print(f"  >>> Avg latency: {result['latency_ms']}ms")
    return result


# ─── 实验6: 边界情况 ──────────────────────────────────

def experiment_edge_cases(mem: MemoryGraph) -> list[dict]:
    """测试系统鲁棒性"""
    print("\n" + "=" * 70)
    print("  [实验6] 边界情况测试")
    print("=" * 70)

    results = []

    # 6.1 空记忆库
    empty_mem = MemoryGraph({"storage_path": "results/empty_test.json"})
    empty_result = empty_mem.search("test", top_k=5)
    ok = empty_result == []
    results.append({"test": "empty_graph", "passed": ok, "detail": f"result={empty_result}"})
    print(f"  6.1 空记忆库: {'✓' if ok else '✗'}")

    # 6.2 相似内容
    mem.store("user", "Python GIL是什么？", topic="python")
    mem.store("user", "解释Python的GIL机制", topic="python")
    r = mem.search("Python全局解释器锁", top_k=3)
    ok = len(r) > 0 and r[0]["score"] > 0.5
    results.append({"test": "similar_content", "passed": ok, "detail": f"top_score={r[0]['score']:.4f}" if r else "no results"})
    print(f"  6.2 相似内容: {'✓' if ok else '✗'} (score={r[0]['score']:.4f})")

    # 6.3 增量存储
    before = len(mem.nodes)
    mem.store("user", "新记忆", topic="test")
    after = len(mem.nodes)
    ok = after == before + 1
    results.append({"test": "incremental_store", "passed": ok, "detail": f"{before}->{after}"})
    print(f"  6.3 增量存储: {'✓' if ok else '✗'} ({before}->{after})")

    # 6.4 主题回忆
    r = mem.recall("rag", top_k=5)
    ok = len(r) > 0 and all(x["topic"] == "rag" for x in r)
    results.append({"test": "topic_recall", "passed": ok, "detail": f"recalled {len(r)} items"})
    print(f"  6.4 主题回忆: {'✓' if ok else '✗'} ({len(r)} items)")

    # 6.5 对称性
    ok = True
    for i in mem.adj_list:
        for j in mem.adj_list[i]:
            if i not in mem.adj_list[j]:
                ok = False
                break
    results.append({"test": "edge_symmetry", "passed": ok})
    print(f"  6.5 边对称性: {'✓' if ok else '✗'}")

    return results


# ─── 运行全部实验 ───────────────────────────────────────

def run_all(output_path: str = "results/experiment_report.json") -> dict:
    """运行全部实验并生成报告"""
    print("╔" + "═" * 68 + "╗")
    print("║  MyGRAGMemory — 完整实验评估                            ║")
    print("╚" + "═" * 68 + "╝")

    mem = MemoryGraph({"storage_path": "results/experiment_memories.json"})

    report = {}
    report["graph_stats"] = experiment_build_graph(mem)
    report["vector_vs_grag"] = experiment_vector_vs_grag(mem)
    report["cross_topic"] = experiment_cross_topic(mem)
    report["graph_analysis"] = experiment_graph_analysis(mem)
    report["ppr_convergence"] = experiment_ppr_convergence(mem)
    report["edge_cases"] = experiment_edge_cases(mem)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 汇总
    ct = report["cross_topic"]
    multi_success = sum(1 for r in ct if r["multi_topic"])
    print(f"\n" + "=" * 70)
    print("  📊 实验报告概要")
    print("=" * 70)
    print(f"  记忆图: {report['graph_stats']['total_memories']}节点, "
          f"{report['graph_stats']['edges']}边, "
          f"{report['graph_stats']['density']}%密度")
    print(f"  跨主题覆盖率: {multi_success}/{len(ct)} "
          f"({multi_success/len(ct)*100:.0f}%)")
    print(f"  PPR收敛: {report['ppr_convergence']['avg_iters']}次迭代平均")
    print(f"  检索延迟: {report['ppr_convergence']['latency_ms']}ms")
    print(f"\n  完整报告: {output_path}")
    return report


if __name__ == "__main__":
    run_all()

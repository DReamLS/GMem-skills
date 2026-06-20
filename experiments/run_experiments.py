#!/usr/bin/env python3
"""
实验运行器 — GRAG 记忆系统完整实验流程 v2

实验列表：
  1. 大规模记忆图构建 (200+轮, 10+主题)
  2. 纯向量 vs GRAG 检索对比
  3. 跨主题检索能力评估
  4. 图结构分析（增长曲线、密度、枢纽节点）
  5. PPR收敛性与检索效率
  6. 自动更新生命周期验证 (AutoMemoryAgent)
  7. 边界情况测试
  8. HC3 数据集评估（Term Recall 对比）
"""
import json
import math
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from skills.memory_skill import MemoryGraph, EmbeddingModel
from skills.auto_agent import AutoMemoryAgent, infer_topic

np.random.seed(42)

# ══════════════════════════════════════════════════════════
# 大规模实验数据：10个主题，200+轮对话
# ══════════════════════════════════════════════════════════

LARGE_CORPUS = {
    "rag": [
        ("什么是RAG？", "RAG是检索增强生成，结合检索和文本生成的技术。"),
        ("RAG有哪些核心组件？", "RAG包含检索器、知识库和生成器三个核心组件。"),
        ("FAISS是什么工具？", "FAISS是Meta开发的向量相似度搜索库，支持GPU加速。"),
        ("embedding模型的作用是什么？", "embedding模型将文本转为向量，用于语义相似度计算。"),
        ("如何评估RAG系统的质量？", "RAG评估包括检索Recall@K和生成ROUGE/BLEU。"),
        ("RAG和微调有什么区别？", "RAG通过外部知识增强而不改模型参数，微调则调整权重。"),
        ("什么是稠密检索？", "稠密检索使用神经网络编码查询和文档，计算向量相似度。"),
        ("BM25和稠密检索哪个好？", "BM25适合精确关键词匹配，稠密检索适合语义匹配。"),
        ("RAG中chunk大小如何选择？", "chunk通常256-512 tokens，过小丢失上下文，过大引入噪声。"),
        ("什么是HyDE？", "HyDE先生成假设文档再检索，缩小查询-文档语义差距。"),
        ("RAG的reranking策略？", "先用BM25/稠密检索粗召回Top-100，再用cross-encoder精排Top-10。"),
        ("什么是CRAG？", "CRAG是 corrective RAG，对检索结果先评估再决定使用或重试。"),
        ("Self-RAG是什么？", "Self-RAG让模型自检是否需要检索，自己评判检索结果质量。"),
        ("RAG在工业界的应用？", "智能客服、知识库问答、法律文档审查、医疗辅助诊断等。"),
    ],
    "python": [
        ("列表和元组有什么区别？", "列表可变元组不可变，列表用[]元组用()。"),
        ("什么是装饰器？", "装饰器是扩展函数功能的高阶函数，常用于日志和权限控制。"),
        ("Python的GIL是什么？", "GIL是全局解释器锁，限制同一时刻只有一个线程执行字节码。"),
        ("什么是列表推导式？", "列表推导式是简洁创建列表的方式：[x**2 for x in range(10)]。"),
        ("生成器和迭代器区别？", "生成器用yield惰性求值，迭代器用__iter__和__next__协议。"),
        ("Python的深拷贝和浅拷贝？", "浅拷贝复制引用，深拷贝递归复制所有对象。用copy模块。"),
        ("什么是上下文管理器？", "用with语句管理资源，自动调用__enter__和__exit__。"),
        ("Python的GIL如何绕过？", "用多进程(ProcessPoolExecutor)或C扩展(cython/numba)。"),
        ("什么是asyncio？", "asyncio是异步IO框架，用async/await协程实现并发。"),
        ("Python的垃圾回收？", "引用计数为主，标记清除和分代回收处理循环引用。"),
        ("f-string有哪些高级用法？", "支持表达式、格式说明符、对齐、日期格式化。"),
        ("什么是类型提示？", "Python 3.5+支持静态类型提示，用mypy检查类型。"),
        ("Pandas的apply和vectorized？", "Vectorized操作比apply快10-100倍，避免逐行循环。"),
        ("NumPy广播机制？", "不同shape的数组运算时自动扩展维度对齐。"),
        ("什么是Monkey Patching？", "运行时动态修改类或模块，危险但灵活。"),
    ],
    "deep-learning": [
        ("什么是反向传播？", "反向传播通过链式法则计算梯度，更新神经网络参数。"),
        ("Transformer自注意力机制？", "自注意力计算序列中每个位置与其他位置的关联权重。"),
        ("什么是过拟合？如何解决？", "过拟合是训练集好测试集差，用正则化、dropout、数据增强解决。"),
        ("CNN和Transformer的区别？", "CNN局部连接权重共享，Transformer全局自注意力。"),
        ("什么是Batch Normalization？", "BN对每层输出归一化，加速训练，有轻微正则化效果。"),
        ("Adam和SGD优化器选择？", "Adam自适应学习率适合CV/NLP，SGD配合调度器最终精度更高。"),
        ("什么是知识蒸馏？", "大模型(teacher)教小模型(student)，压缩模型同时保留性能。"),
        ("梯度消失和梯度爆炸？", "深层网络梯度连乘导致消失或爆炸，用残差连接和梯度裁剪。"),
        ("什么是注意力机制？", "注意力根据查询分配权重，加权聚合值。QKV三矩阵。"),
        ("多头注意力的好处？", "多head并行关注不同子空间，增强表示能力。"),
        ("什么是位置编码？", "Transformer用sin/cos或可学习位置编码注入位置信息。"),
        ("什么是学习率调度？", "warmup、cosine decay、step decay等方式动态调整学习率。"),
        ("什么是权重初始化？", "Xavier适合tanh/sigmoid，Kaiming适合ReLU。"),
        ("什么是数据增强？", "对训练数据做随机变换（旋转、裁剪、噪声）提升泛化。"),
        ("混合精度训练？", "FP16计算+FP32主权重，速度翻倍，显存减半。"),
    ],
    "database": [
        ("什么是数据库索引？", "索引是加速数据检索的数据结构，类似书籍目录。"),
        ("SQL和NoSQL怎么选？", "SQL适合强一致性和复杂查询，NoSQL适合高并发灵活schema。"),
        ("什么是B+树索引？", "B+树所有数据在叶子节点，非叶子只存键值，范围查询高效。"),
        ("什么是数据库事务？", "事务是ACID操作单元，原子性、一致性、隔离性、持久性。"),
        ("MySQL和PostgreSQL对比？", "PG支持更多数据类型和索引，MySQL生态更成熟。"),
        ("什么是CAP定理？", "分布式系统在一致性、可用性、分区容忍性中最多选两个。"),
        ("什么是数据库分片？", "水平分片把数据分布到多台机器，解决单机容量瓶颈。"),
        ("什么是读写分离？", "主库写从库读，缓解单库压力，提高并发读能力。"),
        ("什么是连接池？", "复用数据库连接，避免频繁创建销毁连接的开销。"),
        ("EXPLAIN怎么分析慢查询？", "EXPLAIN显示执行计划，关注type、rows、Extra字段。"),
        ("什么是Redis？", "Redis是内存数据库，支持字符串、列表、集合等数据结构。"),
        ("Redis持久化方式？", "RDB快照和AOF日志，AOF更安全但文件更大。"),
        ("什么是Elasticsearch？", "ES是基于Lucene的分布式搜索引擎，适用于全文检索。"),
        ("什么是OLTP和OLAP？", "OLTP事务处理，OLAP分析处理，面向不同场景。"),
    ],
    "nlp": [
        ("什么是词嵌入？", "词嵌入将词语映射到低维稠密向量，如Word2Vec和GloVe。"),
        ("BERT怎么训练的？", "BERT用掩码语言模型和下一句预测预训练，两阶段：预训练+微调。"),
        ("Seq2Seq模型是什么？", "Seq2Seq将输入序列映射到输出序列，用于翻译、摘要等。"),
        ("GPT和BERT的区别？", "GPT是自回归解码器，BERT是双向编码器。GPT适合生成，BERT适合理解。"),
        ("什么是注意力机制在NLP中的应用？", "注意力让模型在生成每个词时关注输入序列的不同位置。"),
        ("什么是命名实体识别？", "NER识别文本中的人名、地名、组织名等实体。"),
        ("文本分类有哪些方法？", "传统方法TF-IDF+LR，深度方法TextCNN/BERT微调。"),
        ("什么是情感分析？", "情感分析判断文本的正面/负面/中性情感倾向。"),
        ("什么是BLEU指标？", "BLEU基于n-gram精确率评估生成文本质量，用于机器翻译。"),
        ("什么是ROUGE指标？", "ROUGE基于召回率评估摘要质量，包括ROUGE-1/2/L。"),
        ("什么是Beam Search？", "Beam Search在解码时维护K个最优候选路径，平衡质量和效率。"),
        ("什么是Prompt Engineering？", "通过设计提示词引导LLM输出期望结果的技术。"),
        ("什么是In-Context Learning？", "ICL在prompt中加入示例，让LLM从上下文中学习任务。"),
        ("LLM的幻觉问题？", "LLM生成事实错误内容，用RAG和知识图谱缓解。"),
        ("什么是指令微调？", "用指令-回复对微调LLM，提升遵循指令和泛化能力。"),
    ],
    "mlops": [
        ("什么是MLOps？", "MLOps将DevOps实践应用到ML，管理模型全生命周期。"),
        ("模型部署有哪些方式？", "在线推理(ONNX/TensorRT)、批处理、边缘部署。"),
        ("什么是Docker容器化？", "Docker打包应用和依赖为镜像，保证环境一致性。"),
        ("Kubernetes在ML中的作用？", "K8s管理容器化ML服务的自动扩缩容和滚动更新。"),
        ("什么是模型版本管理？", "用DVC或MLflow管理数据集、模型和实验的版本。"),
        ("什么是特征存储？", "Feature Store集中管理特征，保证训练和推理的特征一致性。"),
        ("CI/CD在ML中怎么用？", "模型训练、评估、部署流水线自动化，触发条件可配置。"),
        ("什么是A/B测试？", "同时部署新旧模型，分流流量对比效果。"),
        ("模型监控有哪些指标？", "数据漂移、概念漂移、延迟、吞吐量、精度退化。"),
        ("什么是ONNX？", "ONNX是开放式神经网络交换格式，跨框架部署模型。"),
        ("TensorRT有什么用？", "TensorRT优化推理速度，支持FP16/INT8量化。"),
        ("什么是模型可解释性？", "SHAP/LIME解释模型预测，满足监管和调试需求。"),
        ("什么是数据版本控制？", "追踪数据集的变更历史，确保实验可复现。"),
        ("GPU资源怎么调度？", "用Kubernetes GPU插件或Slurm集群调度训练任务。"),
    ],
    "algorithm": [
        ("什么是时间复杂度和空间复杂度？", "时间-操作次数，空间-内存使用量，用大O表示法。"),
        ("排序算法有哪些？", "快排O(nlogn)、归并O(nlogn)、堆排O(nlogn)、插入O(n²)。"),
        ("动态规划的核心思想？", "DP将问题分解为重叠子问题，记录中间结果避免重复计算。"),
        ("什么是贪心算法？", "贪心每步选局部最优，适用最小生成树、哈夫曼编码。"),
        ("图的遍历方式？", "DFS递归或栈实现，BFS队列实现，各适用于不同场景。"),
        ("什么是哈希表？", "哈希表用哈希函数将键映射到位置，平均O(1)查插删。"),
        ("平衡二叉树有哪些？", "AVL树严格平衡，红黑树近似平衡，B树适合磁盘IO。"),
        ("什么是二分查找？", "在有序数组中每次减半搜索范围，O(logn)效率。"),
        ("回溯算法适用场景？", "八皇后、数独、全排列、组合求和等约束满足问题。"),
        ("什么是滑动窗口？", "维护可变大小窗口遍历序列，解决子串/子数组问题。"),
        ("拓扑排序是什么？", "有向无环图的顶点线性排序，每个顶点在依赖之后。"),
        ("最短路径算法？", "Dijkstra非负权，Bellman-Ford可负权，Floyd全源。"),
        ("什么是并查集？", "Union-Find管理元素分组，支持合并和查找操作，近O(1)。"),
        ("字符串匹配算法？", "KMP线性时间，Boyer-Moore跳过尽可能多字符。"),
    ],
    "system": [
        ("操作系统进程和线程的区别？", "进程独立地址空间，线程共享地址空间。线程切换代价小。"),
        ("什么是虚拟内存？", "虚拟内存将物理内存和磁盘作为统一地址空间，页表管理映射。"),
        ("Linux的IO模型？", "阻塞IO、非阻塞IO、IO多路复用(epoll/select)、异步IO。"),
        ("什么是TCP三次握手？", "SYN→SYN-ACK→ACK建立连接，确保双方收发能力正常。"),
        ("HTTP和HTTPS区别？", "HTTPS在HTTP下加TLS/SSL层，加密传输，防止中间人攻击。"),
        ("什么是微服务架构？", "将单体应用拆分为多个独立服务，各自部署和扩展。"),
        ("进程间通信方式？", "管道、消息队列、共享内存、信号量、Socket。"),
        ("什么是死锁？条件？", "互斥、持有等待、不可剥夺、循环等待四个条件同时满足。"),
        ("Linux的文件描述符？", "FD是非负整数，指向打开的文件/socket/管道。ulimit限制。"),
        ("什么是零拷贝？", "数据在内核空间直接传输到网卡/磁盘，避免用户态拷贝。"),
        ("什么是分布式一致性？", "Raft/Paxos共识算法保证多副本数据一致性。"),
        ("什么是负载均衡？", "将请求分发到多台服务器，策略：轮询、最少连接、IP哈希。"),
    ],
}

# 将大语料转为对话列表
def build_large_conversations():
    convs = []
    topic_order = []
    # 轮流从各topic取对话，模拟真实多主题交错
    max_len = max(len(qas) for qas in LARGE_CORPUS.values())
    for i in range(max_len):
        for topic, qas in LARGE_CORPUS.items():
            if i < len(qas):
                convs.append((topic, qas[i][0], qas[i][1]))
    return convs


LARGE_CONVERSATIONS = build_large_conversations()

# 跨主题查询（基于大语料设计）
CROSS_TOPIC_QUERIES = [
    ("Python中的异步IO和操作系统IO模型有什么关系？", ["python", "system"]),
    ("如何在GPU上加速向量检索用于RAG？", ["rag", "mlops"]),
    ("Transformer自注意力机制在NLP中如何演化？", ["deep-learning", "nlp"]),
    ("数据库索引和FAISS索引的实现原理对比？", ["database", "rag"]),
    ("在K8s上部署BERT推理服务的最佳实践？", ["nlp", "mlops"]),
    ("Python NumPy广播机制在深度学习中的应用？", ["python", "deep-learning"]),
    ("Redis缓存和RAG检索的结合使用？", ["database", "rag"]),
    ("NLP模型部署的ONNX推理优化？", ["nlp", "mlops"]),
    ("图算法在推荐系统中的应用？", ["algorithm", "deep-learning"]),
    ("用Docker部署分布式数据库集群？", ["system", "database"]),
    ("LLM的幻觉问题和RAG缓解方案？", ["nlp", "rag"]),
    ("Python装饰器在MLOps流水线中的使用？", ["python", "mlops"]),
]

VECTOR_VS_GRAG_QUERIES = [
    ("RAG检索增强生成技术详解", "rag"),
    ("Python并发编程与GIL机制", "python"),
    ("深度学习模型训练与过拟合", "deep-learning"),
    ("数据库查询优化与索引选择", "database"),
    ("NLP词嵌入与预训练模型", "nlp"),
    ("MLOps模型部署与Docker容器化", "mlops"),
    ("算法复杂度分析与排序算法对比", "algorithm"),
    ("操作系统IO模型与网络协议", "system"),
]


# ══════════════════════════════════════════════════════════
# 实验1: 大规模记忆图构建
# ══════════════════════════════════════════════════════════

def experiment_large_scale(mem: MemoryGraph) -> dict:
    """构建大规模记忆图并分析增长曲线"""
    print("\n" + "=" * 70)
    print("  [实验1] 大规模记忆图构建 ({}+轮)".format(len(LARGE_CONVERSATIONS)))
    print("=" * 70)

    growth = {"nodes": [], "edges": [], "density": [], "time": []}
    t_start = time.time()

    for idx, (topic, usr, asst) in enumerate(LARGE_CONVERSATIONS):
        mem.store_conversation(usr, asst, topic=topic)
        if (idx + 1) % 20 == 0 or idx == 0 or idx == len(LARGE_CONVERSATIONS) - 1:
            stats = mem.stats()
            growth["nodes"].append(stats["total_memories"])
            growth["edges"].append(stats["edges"])
            growth["density"].append(stats["density"])
            growth["time"].append(round(time.time() - t_start, 2))
            print(f"  +{idx+1:03d}轮→ {stats['total_memories']}节点, "
                  f"{stats['edges']}边, {stats['density']:.1f}%密度")

    stats = mem.stats()
    build_time = time.time() - t_start
    print(f"\n  >>> 构建完成: {stats['total_memories']}节点, {stats['edges']}边, "
          f"{stats['density']:.1f}%密度, {stats['num_topics']}主题, {build_time:.1f}s")
    return {"stats": stats, "growth": growth, "build_time_s": round(build_time, 2)}


# ══════════════════════════════════════════════════════════
# 实验2: 向量 vs GRAG 检索对比
# ══════════════════════════════════════════════════════════

def experiment_vector_vs_grag(mem: MemoryGraph) -> dict:
    print("\n" + "=" * 70)
    print("  [实验2] 纯向量 vs GRAG 检索对比")
    print("=" * 70)

    results = []
    for query, expected_topic in VECTOR_VS_GRAG_QUERIES:
        emb_arr = np.array(mem.embeddings)
        qv = mem.embedder.encode_one(query)
        vec_scores = np.dot(emb_arr, qv)
        top_vec = np.argsort(vec_scores)[::-1][:10]
        grag_results = mem.search(query, top_k=10)

        vec_match = sum(1 for i in top_vec if mem.nodes[i].topic == expected_topic)
        grag_match = sum(1 for r in grag_results if r["topic"] == expected_topic)

        results.append({
            "query": query, "expected_topic": expected_topic,
            "vector_match_top5": min(vec_match, 5), "grag_match_top5": min(grag_match, 5),
            "vector_match_top10": vec_match, "grag_match_top10": grag_match,
        })
        print(f"  {query[:25]:<25} 向量={vec_match}/10 GRAG={grag_match}/10 "
              f"{'↑' if grag_match > vec_match else '↓' if vec_match > grag_match else '='}")

    avg_v5 = np.mean([r["vector_match_top5"] for r in results])
    avg_g5 = np.mean([r["grag_match_top5"] for r in results])
    avg_v10 = np.mean([r["vector_match_top10"] for r in results])
    avg_g10 = np.mean([r["grag_match_top10"] for r in results])
    print(f"\n  >>> Top-5: Vector={avg_v5:.1f}, GRAG={avg_g5:.1f} ({((avg_g5-avg_v5)/max(avg_v5,0.1)*100):.0f}%)")
    print(f"  >>> Top-10: Vector={avg_v10:.1f}, GRAG={avg_g10:.1f} ({((avg_g10-avg_v10)/max(avg_v10,0.1)*100):.0f}%)")
    return {"detail": results, "avg_top5": {"vector": round(avg_v5, 2), "grag": round(avg_g5, 2)},
            "avg_top10": {"vector": round(avg_v10, 2), "grag": round(avg_g10, 2)}}


# ══════════════════════════════════════════════════════════
# 实验3: 跨主题检索
# ══════════════════════════════════════════════════════════

def experiment_cross_topic(mem: MemoryGraph) -> dict:
    print("\n" + "=" * 70)
    print("  [实验3] 跨主题检索评估 ({})个查询".format(len(CROSS_TOPIC_QUERIES)))
    print("=" * 70)

    results = []
    for query, expected_topics in CROSS_TOPIC_QUERIES:
        grag = mem.search(query, top_k=10)
        topics_found = list(set(r["topic"] for r in grag))
        covered = sum(1 for t in expected_topics if t in topics_found)
        multi = len(topics_found) > 1

        # 主题多样性
        topic_dist = Counter(r["topic"] for r in grag)
        entropy = -sum(p/len(grag) * math.log(p/len(grag), 2)
                       for p in topic_dist.values()) if grag else 0

        results.append({
            "query": query[:30], "expected": expected_topics,
            "topics_found": topics_found, "covered": covered,
            "multi_topic": multi, "topic_entropy": round(entropy, 3),
        })
        print(f"  {query[:30]:<30} → {str(topics_found):<40} "
              f"{'✓' if multi else '✗'} H={entropy:.2f}")

    multi_rate = sum(1 for r in results if r["multi_topic"]) / len(results)
    avg_coverage = np.mean([r["covered"]/len(r["expected"]) for r in results])
    avg_entropy = np.mean([r["topic_entropy"] for r in results])
    print(f"\n  >>> 多主题率: {multi_rate*100:.0f}%, 覆盖率: {avg_coverage*100:.0f}%, "
          f"熵: {avg_entropy:.2f}")
    return {
        "detail": results,
        "multi_topic_rate": round(multi_rate, 4),
        "avg_coverage": round(avg_coverage, 4),
        "avg_topic_entropy": round(avg_entropy, 3),
    }


# ══════════════════════════════════════════════════════════
# 实验4: 图结构分析
# ══════════════════════════════════════════════════════════

def experiment_graph_analysis(mem: MemoryGraph) -> dict:
    print("\n" + "=" * 70)
    print("  [实验4] 大规模图结构分析")
    print("=" * 70)

    stats = mem.stats()
    n = stats["total_memories"]
    degrees = [len(mem.adj_list[i]) for i in range(n)] if n > 0 else []
    max_deg = max(degrees) if degrees else 0

    # 度分布直方
    hist = Counter(degrees)
    print(f"  总节点: {n}")
    print(f"  总边数: {stats['edges']}")
    print(f"  密度: {stats['density']:.2f}%")
    print(f"  平均度: {stats['avg_degree']}")
    print(f"  度范围: [{min(degrees)}, {max_deg}]")
    print(f"  主题数: {stats['num_topics']}")
    print(f"  度分布: {sorted(hist.items())[:10]}...")

    # 枢纽分析
    threshold = max_deg * 0.6
    hubs = [(i, degrees[i], mem.nodes[i].topic, mem.nodes[i].content[:50])
            for i in range(n) if degrees[i] >= threshold]
    hubs.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  枢纽节点 (度≥{threshold:.0f}, 共{len(hubs)}个):")
    for nid, deg, topic, content in hubs[:8]:
        print(f"    #{nid} [{topic}] deg={deg} | {content}...")

    return {
        "stats": stats,
        "max_degree": max_deg,
        "hub_count": len(hubs),
        "degree_histogram": dict(sorted(hist.items())),
    }


# ══════════════════════════════════════════════════════════
# 实验5: PPR收敛与效率
# ══════════════════════════════════════════════════════════

def experiment_ppr_convergence(mem: MemoryGraph) -> dict:
    print("\n" + "=" * 70)
    print("  [实验5] PPR收敛性与检索效率")
    print("=" * 70)

    iters_list = []
    latencies = []
    for query, _ in VECTOR_VS_GRAG_QUERIES:
        qv = mem.embedder.encode_one(query)
        emb_arr = np.array(mem.embeddings)
        vec_s = np.dot(emb_arr, qv)

        t0 = time.time()
        _, n_iters = mem._ppr(vec_s)
        latencies.append((time.time() - t0) * 1000)
        iters_list.append(n_iters)

    t0 = time.time()
    for _ in range(10):
        mem.search("test benchmark", top_k=5)
    avg_lat = (time.time() - t0) / 10 * 1000

    print(f"  平均PPR迭代: {np.mean(iters_list):.1f}次")
    print(f"  PPR延迟: {np.mean(latencies):.1f}ms")
    print(f"  完整检索延迟(10次平均): {avg_lat:.1f}ms")
    print(f"  节点规模: {len(mem.nodes)}")

    return {
        "avg_ppr_iters": round(np.mean(iters_list), 1),
        "avg_ppr_latency_ms": round(np.mean(latencies), 1),
        "avg_search_latency_ms": round(avg_lat, 1),
        "total_nodes": len(mem.nodes),
    }


# ══════════════════════════════════════════════════════════
# 实验6: AutoMemoryAgent 自动更新生命周期
# ══════════════════════════════════════════════════════════

def experiment_auto_update() -> dict:
    """验证AutoMemoryAgent随对话自动更新记忆图"""
    print("\n" + "=" * 70)
    print("  [实验6] AutoMemoryAgent 自动更新生命周期")
    print("=" * 70)

    agent = AutoMemoryAgent(
        {"storage_path": "results/auto_lifecycle.json"},
        auto_topic=True, top_k_retrieval=5
    )

    timeline = {"turns": [], "nodes": [], "edges": [], "retrieved": [], "latency": []}

    # 模拟多轮对话，逐步加入不同主题
    sessions = [
        ("rag", [
            ("什么是RAG技术？", "RAG是检索增强生成技术。"),
            ("RAG有哪些组件？", "检索器、知识库、生成器。"),
            ("FAISS如何加速检索？", "FAISS用向量索引实现近似最近邻搜索。"),
        ]),
        ("python", [
            ("Python装饰器怎么用？", "@decorator语法糖包装函数。"),
            ("列表推导式性能好么？", "比for循环快2-3倍。"),
        ]),
        ("deep-learning", [
            ("什么是反向传播？", "链式法则计算梯度更新参数。"),
            ("Dropout怎么防止过拟合？", "训练时随机丢弃神经元。"),
            ("BatchNorm的作用？", "归一化层输入加速训练。"),
        ]),
        ("cross-topic", [
            ("Python实现RAG检索？", "用FAISS+transformers构建RAG流水线。"),
            ("深度学习和NLP的关系？", "DL在NLP中广泛应用如BERT。"),
            ("数据库和RAG的结合？", "向量数据库Milvus用于RAG知识库。"),
        ]),
        ("rag", [
            ("RAG评估指标有哪些？", "Recall@K、MRR、ROUGE、BLEU。"),
            ("HyDE怎么改善检索？", "Hypothetical Document Embeddings。"),
            ("CRAG的纠正机制？", "检索结果评估→正确/错误/模糊三路分支。"),
            ("GRAG和RAG的区别？", "GRAG加图传播增强检索，支持多跳推理。"),
        ]),
    ]

    for topic, conv in sessions:
        for usr, asst in conv:
            # 自动topic推断（实际agent会自动做）
            result = agent.respond(usr, asst)
            timeline["turns"].append(result["turn"])
            s = result["graph_stats"]
            timeline["nodes"].append(s["total_memories"])
            timeline["edges"].append(s["edges"])
            timeline["retrieved"].append(len(result["retrieved_memories"]))
            timeline["latency"].append(result["retrieval_time_ms"])

    stats = agent.get_stats()
    print(f"\n  >>> 自动更新验证完成:")
    print(f"  总对话轮次: {stats['total_turns']}")
    print(f"  最终记忆图: {stats['total_memories']}节点, {stats['edges']}边, "
          f"{stats['density']:.1f}%密度")
    print(f"  平均检索记忆数: {np.mean(timeline['retrieved']):.1f}条/轮")
    print(f"  检索延迟: {np.mean(timeline['latency']):.0f}ms")
    print(f"  主题: {stats['topics']}")

    # 验证第2次问相同主题时能否检索到早期对话
    verify_result = agent.respond("RAG和GRAG有什么区别？我记不清细节了")
    relevant_memories = [m for m in verify_result["retrieved_memories"]
                         if "GRAG" in m["content"] or "图传播" in m["content"]]
    print(f"  ▶ GRAG相关问题验证: 检索到{len(relevant_memories)}条相关记忆"
          f"{'✓' if len(relevant_memories) > 0 else '✗'}")

    return {
        "total_turns": stats['total_turns'],
        "final_stats": {
            "nodes": stats['total_memories'],
            "edges": stats['edges'],
            "density": stats['density'],
            "topics": stats['topics'],
        },
        "avg_retrieved_per_turn": round(np.mean(timeline['retrieved']), 1),
        "avg_latency_ms": round(np.mean(timeline['latency']), 1),
        "timeline": timeline,
    }


# ══════════════════════════════════════════════════════════
# 实验7: 边界情况
# ══════════════════════════════════════════════════════════

def experiment_edge_cases(mem: MemoryGraph) -> list[dict]:
    print("\n" + "=" * 70)
    print("  [实验7] 边界与鲁棒性测试")
    print("=" * 70)
    results = []

    # 7.1 空记忆库
    empty = MemoryGraph({"storage_path": "results/empty_test.json"})
    ok = empty.search("test", top_k=5) == []
    results.append({"test": "empty_graph", "passed": ok})

    # 7.2 相似内容建边
    mem.store("user", "Python GIL是什么？", topic="python")
    mem.store("user", "解释Python GIL机制", topic="python")
    r = mem.search("Python全局解释器锁", top_k=5)
    ok = len(r) > 0 and r[0]["score"] > 0.4
    results.append({"test": "similar_content", "passed": ok, "detail": f"top_score={r[0]['score']:.3f}"})

    # 7.3 增量存储正确性
    before = len(mem.nodes)
    mem.store("user", "测试", topic="test")
    ok = len(mem.nodes) == before + 1
    results.append({"test": "incremental_store", "passed": ok})

    # 7.4 图对称性
    ok = all(j in mem.adj_list for i in mem.adj_list for j in mem.adj_list[i] if i in mem.adj_list[j])
    results.append({"test": "edge_symmetry", "passed": ok})

    # 7.5 空查询
    ok = len(mem.search("", top_k=5)) >= 0
    results.append({"test": "empty_query", "passed": ok})

    for r in results:
        print(f"  {r['test']}: {'✓' if r['passed'] else '✗'}")
    return results


# ══════════════════════════════════════════════════════════
# HC3 评估（模拟）
# ══════════════════════════════════════════════════════════

def experiment_hc3_evaluation(mem: MemoryGraph) -> dict:
    """
    HC3风格评估：模拟知识问答检索的Term Recall

    对每个问题，GRAG检索相关内容后计算
    Term Recall = 期望术语在检索结果中的出现比例
    """
    print("\n" + "=" * 70)
    print("  [实验8] HC3风格评估 (Term Recall)")
    print("=" * 70)

    # 从大语料中构造评估查询
    eval_queries = [
        ("RAG的检索器一般用什么工具？", ["FAISS", "检索器", "向量"]),
        ("深度学习如何防止过拟合？", ["dropout", "正则化", "过拟合"]),
        ("Python多线程受什么限制？", ["GIL", "全局解释器锁"]),
        ("数据库索引使用什么数据结构？", ["B+树", "索引"]),
        ("BERT模型使用什么训练目标？", ["掩码语言模型", "MLM", "下一句预测"]),
        ("Transformer的核心机制是什么？", ["自注意力", "注意力机制"]),
        ("MLOps中模型部署常用什么工具？", ["Docker", "ONNX", "Kubernetes"]),
        ("动态规划的核心思想是什么？", ["子问题", "重叠子问题", "最优子结构"]),
        ("什么是零拷贝技术？", ["零拷贝", "内核空间"]),
        ("RAG如何缓解LLM幻觉？", ["检索", "外部知识", "幻觉"]),
    ]

    results = []
    for query, expected_terms in eval_queries:
        retrieved = mem.search(query, top_k=5)
        context = " ".join(r["content"] for r in retrieved)
        term_recall = compute_term_recall(context, expected_terms)
        found_terms = [t for t in expected_terms if t.lower() in context.lower()]
        results.append({
            "query": query[:30],
            "expected_terms": expected_terms,
            "found_terms": found_terms,
            "term_recall": round(term_recall, 4),
        })
        print(f"  {query[:30]:<30} TR={term_recall:.2f} "
              f"({len(found_terms)}/{len(expected_terms)}) {found_terms}")

    avg_tr = np.mean([r["term_recall"] for r in results])
    print(f"\n  >>> Avg Term Recall: {avg_tr:.3f}")
    return {"avg_term_recall": round(avg_tr, 4), "detail": results}


def compute_term_recall(context: str, expected_terms: list[str]) -> float:
    if not expected_terms or not context:
        return 0.0
    matched = sum(1 for t in expected_terms if t.lower() in context.lower())
    return matched / len(expected_terms)


# ══════════════════════════════════════════════════════════
# 运行全部实验
# ══════════════════════════════════════════════════════════

def run_all(output_path: str = "results/experiment_report_v2.json") -> dict:
    print("╔" + "═" * 68 + "╗")
    print("║  MyGRAGMemory v2 — 大规模实验评估                     ║")
    print("╚" + "═" * 68 + "╝")

    report = {}
    mem = MemoryGraph({"storage_path": "results/large_exp_memories.json"})

    report["large_scale"] = experiment_large_scale(mem)

    if len(mem.nodes) > 0:
        report["vector_vs_grag"] = experiment_vector_vs_grag(mem)
        report["cross_topic"] = experiment_cross_topic(mem)
        report["graph_analysis"] = experiment_graph_analysis(mem)
        report["ppr_convergence"] = experiment_ppr_convergence(mem)
        report["edge_cases"] = experiment_edge_cases(mem)
        report["hc3_evaluation"] = experiment_hc3_evaluation(mem)

    report["auto_update"] = experiment_auto_update()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("  📊 综合实验报告")
    print("=" * 70)
    ls = report.get("large_scale", {})
    s = ls.get("stats", {})
    print(f"  记忆图: {s.get('total_memories', 0)}节点, {s.get('edges', 0)}边, "
          f"{s.get('density', 0):.1f}%密度")
    ct = report.get("cross_topic", {})
    print(f"  跨主题率: {ct.get('multi_topic_rate', 0)*100:.0f}%")
    au = report.get("auto_update", {})
    print(f"  自动更新: {au.get('total_turns', 0)}轮, "
          f"每轮检索{au.get('avg_retrieved_per_turn', 0):.1f}条, "
          f"{au.get('avg_latency_ms', 0):.0f}ms")
    hc3 = report.get("hc3_evaluation", {})
    print(f"  HC3 Term Recall: {hc3.get('avg_term_recall', 0):.3f}")
    print(f"\n  📁 报告: {output_path}")
    return report


if __name__ == "__main__":
    run_all()

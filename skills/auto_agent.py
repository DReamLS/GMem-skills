#!/usr/bin/env python3
"""
AutoMemoryAgent: 自动对话记忆智能体

核心特性：
1. 自动存储 — 每轮对话自动存入记忆图
2. 自动检索 — 每次回答前自动GRAG检索相关记忆
3. 上下文增强 — 检索结果拼接为上下文提示
4. 实时图更新 — 图随对话动态增长，增量更新

与普通 MemoryGraph 的区别：
  - MemoryGraph: 手动存储 + 手动检索（固定数据库）
  - AutoMemoryAgent: 自动存储 + 自动检索 + 实时图更新（动态增长）

用法：
  agent = AutoMemoryAgent()
  agent.respond("什么是RAG?")  # 自动检索→生成→存储
"""
import json
import os
import sys
import time
from collections import Counter
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skills.memory_skill import MemoryGraph

# ─── 简易话题推断 ────────────────────────────────────────

TOPIC_KEYWORDS = {
    "rag": ["rag", "检索增强", "retrieval", "faiss", "向量检索", "召回", "rerank",
            "知识库", "index", "embedding", "dense", "sparse", "检索"],
    "python": ["python", "列表", "元组", "装饰器", "gil", "线程", "协程",
               "异步", "生成器", "迭代器", "lambda", "pandas", "numpy"],
    "deep-learning": ["深度学习", "神经网络", "反向传播", "transformer", "attention",
                      "自注意力", "cnn", "rnn", "lstm", "过拟合", "dropout",
                      "batch norm", "梯度", "激活函数", "损失函数"],
    "database": ["数据库", "索引", "sql", "nosql", "mysql", "postgresql",
                 "redis", "mongodb", "事务", "锁", "查询优化", "连接"],
    "nlp": ["nlp", "自然语言", "词嵌入", "bert", "gpt", "llm", "大模型",
            "seq2seq", "机器翻译", "文本分类", "情感分析", "命名实体"],
    "mlops": ["mlops", "部署", "docker", "k8s", "kubernetes", "ci/cd",
              "模型服务", "推理", "onnx", "tensorrt", "流水线"],
    "algorithm": ["算法", "复杂度", "排序", "搜索", "动态规划", "贪心",
                  "图论", "树", "栈", "队列", "链表", "哈希"],
    "system": ["操作系统", "内存", "进程", "文件系统", "网络协议",
               "tcp/ip", "http", "并发", "调度", "io"],
}


def infer_topic(text: str) -> str:
    """根据关键词推断文本所属主题"""
    text_lower = text.lower()
    scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        scores[topic] = sum(1 for kw in keywords if kw.lower() in text_lower)
    if max(scores.values()) == 0:
        return "general"
    return max(scores, key=scores.get)


# ══════════════════════════════════════════════════════════
# AutoMemoryAgent
# ══════════════════════════════════════════════════════════

class AutoMemoryAgent:
    """
    自动对话记忆智能体

    每次 respond() 自动执行：
      1. 推断用户查询主题
      2. GRAG检索相关历史记忆
      3. 将检索结果作为上下文提示
      4. 存储用户查询和助手回复到记忆图
      5. 图自动增量更新

    Args:
        memory_config: 传给 MemoryGraph 的配置 dict
        auto_topic: 是否自动推断主题（否则用 "general"）
        top_k_retrieval: 每次检索返回的记忆数
        system_prompt: 系统提示词模板
    """

    def __init__(
        self,
        memory_config: Optional[dict] = None,
        auto_topic: bool = True,
        top_k_retrieval: int = 5,
        system_prompt: Optional[str] = None,
    ):
        cfg = memory_config or {}
        cfg.setdefault("storage_path", "results/auto_memories.json")
        self.memory = MemoryGraph(cfg)

        self.auto_topic = auto_topic
        self.top_k = top_k_retrieval
        self.system_prompt = system_prompt or (
            "You are a helpful assistant with memory.\n"
            "Below are relevant past conversation memories retrieved by GRAG.\n"
            "Use them to provide contextually informed responses.\n"
        )

        # 对话历史计数
        self._turn_count = 0
        self._last_topic = "general"

    # ─── 核心方法 ─────────────────────────────────────────

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """对外暴露的GRAG检索接口"""
        return self.memory.search(query, top_k=top_k or self.top_k)

    def respond(self, user_msg: str, assistant_msg: Optional[str] = None,
                topic: Optional[str] = None) -> dict:
        """
        处理一轮对话（自动存储+检索）

        Args:
            user_msg: 用户消息
            assistant_msg: 助手回复（若None则用模拟回复）
            topic: 主题（若None则自动推断）

        Returns:
            dict: 含检索结果、图状态等信息
        """
        self._turn_count += 1

        # 1. 推断主题
        if topic is None and self.auto_topic:
            topic = infer_topic(user_msg)
            # 检查与历史记忆的关联度
            if self.memory.nodes:
                results = self.memory.search(user_msg, top_k=3)
                if results:
                    # 如果检索到的记忆主题一致，沿用该主题
                    top_topics = [r["topic"] for r in results[:3]]
                    from collections import Counter
                    topic_counts = Counter(top_topics)
                    retrieved_topic = topic_counts.most_common(1)[0][0]
                    if retrieved_topic != "general":
                        topic = retrieved_topic
        elif topic is None:
            topic = "general"

        self._last_topic = topic

        # 2. GRAG检索相关记忆（在生成回复之前）
        retrieval_start = time.time()
        memories = self.memory.search(user_msg, top_k=self.top_k)
        retrieval_time = (time.time() - retrieval_start) * 1000

        # 3. 构建增强上下文
        context = self._build_context(memories)

        # 4. 模拟回复（实际使用时接入LLM）
        if assistant_msg is None:
            assistant_msg = self._simulate_response(user_msg, memories)

        # 5. 自动存储到记忆图
        store_start = time.time()
        uid = self.memory.store("user", user_msg, topic=topic)
        aid = self.memory.store("assistant", assistant_msg, topic=topic)
        store_time = (time.time() - store_start) * 1000

        stats = self.memory.stats()

        return {
            "turn": self._turn_count,
            "topic": topic,
            "user_msg": user_msg,
            "assistant_msg": assistant_msg,
            "retrieved_memories": memories,
            "retrieval_time_ms": round(retrieval_time, 1),
            "store_time_ms": round(store_time, 1),
            "node_ids": (uid, aid),
            "graph_stats": stats,
        }

    def _build_context(self, memories: list[dict]) -> str:
        """构建检索增强的上下文提示"""
        if not memories:
            return ""
        parts = ["[Retrieved Relevant Memories]"]
        for i, m in enumerate(memories, 1):
            role_tag = "User" if m["role"] == "user" else "Assistant"
            parts.append(
                f"  #{i} [{m['topic']}] ({role_tag}, "
                f"score={m['score']:.3f}): {m['content']}"
            )
        return "\n".join(parts)

    def _simulate_response(self, query: str, memories: list[dict]) -> str:
        """
        模拟回复（基于检索到的记忆构建上下文感知回复）
        实际使用时接入真实LLM
        """
        # 简化的回复策略：根据检索结果构建回复
        retrieved_topics = set(m["topic"] for m in memories)
        has_history = len(memories) > 0

        if has_history:
            top_memory = memories[0]
            topic_info = f" (related to {top_memory['topic']})"
        else:
            topic_info = ""

        return (
            f"[Auto response to: {query[:50]}...]{topic_info} "
            f"(retrieved {len(memories)} relevant memories "
            f"from topics: {', '.join(retrieved_topics) if retrieved_topics else 'none'})"
        )

    # ─── 对话会话管理 ─────────────────────────────────────

    def chat_session(self, conversation: list[tuple[str, str]],
                     topics: Optional[list[str]] = None) -> list[dict]:
        """
        模拟多轮对话会话

        Args:
            conversation: [(user_msg, assistant_msg), ...]
            topics: 每轮的主题（None则自动推断）

        Returns:
            list[dict]: 每轮的 respond() 结果
        """
        results = []
        for i, (user_msg, assistant_msg) in enumerate(conversation):
            topic = topics[i] if topics and i < len(topics) else None
            result = self.respond(user_msg, assistant_msg, topic=topic)
            results.append(result)
            # 实时输出状态
            stats = result["graph_stats"]
            self._print_turn(result)
        return results

    def _print_turn(self, result: dict):
        """打印单轮对话状态"""
        stats = result["graph_stats"]
        mems = result["retrieved_memories"]
        print(
            f"  T{result['turn']:03d} [{result['topic']:<12}] "
            f"→ 图:{stats['total_memories']}节点/{stats['edges']}边 "
            f"| GRAG检索{len(mems)}条 "
            f"({result['retrieval_time_ms']:.0f}ms)"
        )

    def get_stats(self) -> dict:
        """获取系统完整统计"""
        stats = self.memory.stats()
        stats["total_turns"] = self._turn_count
        # 检索延迟基准
        if self.memory.nodes:
            t0 = time.time()
            _ = self.memory.search("test benchmark", top_k=5)
            stats["avg_search_latency_ms"] = round((time.time() - t0) * 1000, 1)
        return stats


# ══════════════════════════════════════════════════════════
# 交互式评测 CLI
# ══════════════════════════════════════════════════════════

def interactive_demo():
    """交互式演示：展示自动更新效果"""
    agent = AutoMemoryAgent({"storage_path": "results/auto_demo.json"})

    print("\n" + "=" * 70)
    print("  AutoMemoryAgent — 自动对话记忆演示")
    print("=" * 70)
    print("  每轮对话将自动:")
    print("    1. 推断主题")
    print("    2. GRAG检索历史相关记忆")
    print("    3. 存储到记忆图（图自动更新）")
    print("  输入 'stats' 查看图状态，'exit' 退出\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input == "exit":
            break
        if user_input == "stats":
            s = agent.get_stats()
            print(f"  [STATS] 记忆图: {s['total_memories']}节点, {s['edges']}边, "
                  f"{s['density']}%密度, {s['num_topics']}主题")
            print(f"         对话轮次: {s['total_turns']}, "
                  f"检索延迟: {s.get('avg_search_latency_ms', 'N/A')}ms")
            continue

        result = agent.respond(user_input)
        stats = result["graph_stats"]
        mem_count = len(result["retrieved_memories"])
        print(f"  [{result['topic']}] "
              f"图→{stats['total_memories']}节点/{stats['edges']}边 "
              f"| GRAG→{mem_count}条记忆 "
              f"({result['retrieval_time_ms']:.0f}ms)")
        if mem_count > 0:
            print(f"  └─ 相关记忆: {result['retrieved_memories'][0]['content'][:70]}...")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AutoMemoryAgent Demo")
    parser.add_argument("--demo", action="store_true", help="交互式演示")
    parser.add_argument("--reset", action="store_true", help="重置记忆")
    args = parser.parse_args()

    if args.reset:
        paths = ["results/auto_memories.json", "results/auto_demo.json"]
        for p in paths:
            if os.path.exists(p):
                os.remove(p)
                print(f"Removed {p}")
    elif args.demo:
        interactive_demo()
    else:
        print("Usage: python -m skills.auto_agent --demo")

#!/usr/bin/env python3
"""
MyGRAGMemory: GRAG-Enhanced Agent Memory Skill

可作为 Claude Code Skill 调用，提供：
  - store / store_conversation: 记忆存储
  - search: GRAG增强检索（向量 + PPR图传播）
  - recall: 按主题回忆
  - stats: 系统统计

用法：
  python -m skills.memory_skill                    # 完整实验
  python -m skills.memory_skill --interactive      # 交互模式
  python -m skills.memory_skill --cmd "q RAG技术"  # 单条命令
"""
import argparse
import json
import os
import time
from typing import Optional

import numpy as np

# ─── 嵌入模型封装 ────────────────────────────────────────

class EmbeddingModel:
    """MiniLM嵌入模型封装（自动降级到随机嵌入）"""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.dim = 384
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            print("[INFO] sentence-transformers not available, using random embeddings")

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.model:
            return self.model.encode(texts, normalize_embeddings=True)
        # 降级：随机384维向量
        rng = np.random.RandomState(42)
        return rng.randn(len(texts), self.dim).astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# ─── 记忆节点 ────────────────────────────────────────────

class MemoryNode:
    """单条对话记忆"""
    def __init__(self, node_id: int, role: str, content: str,
                 topic: str = "general", timestamp: Optional[float] = None):
        self.id = node_id
        self.role = role
        self.content = content
        self.topic = topic
        self.timestamp = timestamp or time.time()
        self.embedding: Optional[np.ndarray] = None

    def to_dict(self):
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "topic": self.topic,
            "timestamp": self.timestamp,
        }


# ══════════════════════════════════════════════════════════
# GRAG 记忆图 — 核心实现
# ══════════════════════════════════════════════════════════

class MemoryGraph:
    """
    GRAG增强的记忆图系统

    图结构：
      - 节点: 对话记忆（含embedding向量）
      - 边1: 语义相似边 (cosine > theta_s)
      - 边2: 同主题边 (same topic + cosine > theta_t)
      - 边3: 时序相邻边 (同主题相邻轮次)

    检索流程：
      1. 查询向量编码 → 余弦相似度 → Personalization Vector
      2. PPR图传播 (迭代法)
      3. 加权融合 → Top-K排序
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.similarity_threshold = cfg.get("similarity_threshold", 0.60)
        self.topic_threshold = cfg.get("topic_threshold", 0.40)
        self.max_edges_per_node = cfg.get("max_edges_per_node", 8)
        self.ppr_alpha = cfg.get("ppr_alpha", 0.85)
        self.ppr_max_iter = cfg.get("ppr_max_iter", 50)
        self.ppr_convergence = cfg.get("ppr_convergence_threshold", 1e-6)
        self.vector_weight = cfg.get("vector_weight", 0.6)
        self.storage_path = cfg.get("storage_path", "results/memories.json")

        self.embedder = EmbeddingModel(cfg.get("embedding_model", "all-MiniLM-L6-v2"))
        self.nodes: list[MemoryNode] = []
        self.embeddings: list[np.ndarray] = []
        self.adj_list: dict[int, set[int]] = {}
        self._counter = 0
        self._load()

    # ─── 持久化 ───────────────────────────────────────────

    def _save(self):
        data = {
            "nodes": [n.to_dict() for n in self.nodes],
            "counter": self._counter,
            "config": {
                "similarity_threshold": self.similarity_threshold,
                "topic_threshold": self.topic_threshold,
                "ppr_alpha": self.ppr_alpha,
                "vector_weight": self.vector_weight,
            },
        }
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for nd in data.get("nodes", []):
                    node = MemoryNode(nd["id"], nd["role"], nd["content"],
                                      nd.get("topic", "general"), nd.get("timestamp"))
                    self.nodes.append(node)
                    self._counter = max(self._counter, nd["id"] + 1)
                self._counter = max(self._counter, data.get("counter", 0))
                if self.nodes:
                    self._reindex()
            except Exception:
                self.nodes = []
                self._counter = 0

    def _reindex(self):
        texts = [n.content for n in self.nodes]
        embs = self.embedder.encode(texts)
        self.embeddings = [embs[i] for i in range(len(texts))]
        self._rebuild_graph()

    # ─── 图构建 ───────────────────────────────────────────

    def _rebuild_graph(self):
        n = len(self.nodes)
        self.adj_list = {i: set() for i in range(n)}
        if n < 2:
            return
        emb = np.array(self.embeddings)
        sim_matrix = np.dot(emb, emb.T)

        for i in range(n):
            sims = [(j, sim_matrix[i][j]) for j in range(n) if j != i]
            sims.sort(key=lambda x: x[1], reverse=True)
            added = 0
            for j, s in sims:
                if s < self.similarity_threshold or added >= self.max_edges_per_node:
                    break
                self.adj_list[i].add(j)
                self.adj_list[j].add(i)
                added += 1

        for i in range(n):
            ti = self.nodes[i].topic
            if not ti:
                continue
            for j in range(i + 1, n):
                if j in self.adj_list[i]:
                    continue
                if ti == self.nodes[j].topic and sim_matrix[i][j] > self.topic_threshold:
                    self.adj_list[i].add(j)
                    self.adj_list[j].add(i)

        topic_groups = {}
        for i, node in enumerate(self.nodes):
            topic_groups.setdefault(node.topic, []).append(i)
        for indices in topic_groups.values():
            for k in range(1, len(indices)):
                self.adj_list[indices[k]].add(indices[k - 1])
                self.adj_list[indices[k - 1]].add(indices[k])

    # ─── 记忆存储 ─────────────────────────────────────────

    def store(self, role: str, content: str, topic: str = "general") -> int:
        """存储单条对话记忆"""
        node = MemoryNode(self._counter, role, content, topic)
        node.embedding = self.embedder.encode_one(content)
        self.nodes.append(node)
        self.embeddings.append(node.embedding)
        self._counter += 1

        n = len(self.nodes)
        self.adj_list[n - 1] = set()
        if n > 1:
            new_emb = node.embedding
            sims = [(i, float(np.dot(self.embeddings[i], new_emb)))
                    for i in range(n - 1)
                    if float(np.dot(self.embeddings[i], new_emb)) > self.similarity_threshold]
            sims.sort(key=lambda x: x[1], reverse=True)
            for i, _ in sims[:self.max_edges_per_node]:
                self.adj_list[i].add(n - 1)
                self.adj_list[n - 1].add(i)

            if topic:
                for i in range(n - 1):
                    if self.nodes[i].topic == topic and i not in self.adj_list[n - 1]:
                        sim = float(np.dot(self.embeddings[i], new_emb))
                        if sim > self.topic_threshold:
                            self.adj_list[i].add(n - 1)
                            self.adj_list[n - 1].add(i)

            for i in range(n - 2, -1, -1):
                if self.nodes[i].topic == topic:
                    self.adj_list[n - 1].add(i)
                    self.adj_list[i].add(n - 1)
                    break

        self._save()
        return node.id

    def store_conversation(self, user_msg: str, assistant_msg: str,
                           topic: str = "general") -> tuple[int, int]:
        uid = self.store("user", user_msg, topic)
        aid = self.store("assistant", assistant_msg, topic)
        return uid, aid

    # ─── PPR 图传播 ───────────────────────────────────────

    def _ppr(self, personalization: np.ndarray) -> tuple[np.ndarray, int]:
        n = len(self.nodes)
        pv = personalization.copy()
        s = pv.sum()
        if s > 0:
            pv = pv / s
        scores = pv.copy()
        n_iters = 0
        for _ in range(self.ppr_max_iter):
            n_iters += 1
            new_scores = np.zeros(n)
            for i in range(n):
                if self.adj_list[i]:
                    ns = sum(scores[j] / max(len(self.adj_list[j]), 1)
                             for j in self.adj_list[i])
                    new_scores[i] = (1 - self.ppr_alpha) * ns
                new_scores[i] += self.ppr_alpha * pv[i]
            diff = float(np.sum(np.abs(new_scores - scores)))
            if diff < self.ppr_convergence:
                break
            scores = new_scores
        return scores, n_iters

    # ─── GRAG 检索 ────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """GRAG检索：向量相似度 + PPR图传播 + 分数融合"""
        n = len(self.nodes)
        if n == 0:
            return []

        qv = self.embedder.encode_one(query)
        emb_arr = np.array(self.embeddings)
        vec_scores = np.dot(emb_arr, qv)
        ppr_scores, n_iters = self._ppr(vec_scores)

        def _norm(x):
            return (x - x.min()) / max(x.max() - x.min(), 1e-8)

        final = self.vector_weight * _norm(vec_scores) + (1 - self.vector_weight) * _norm(ppr_scores)
        top_idx = np.argsort(final)[::-1][:top_k]

        results = []
        for idx in top_idx:
            node = self.nodes[int(idx)]
            results.append({
                **node.to_dict(),
                "score": round(float(final[idx]), 4),
                "vector_score": round(float(vec_scores[idx]), 4),
                "ppr_score": round(float(ppr_scores[idx]), 4),
                "ppr_iters": n_iters,
            })
        return results

    def recall(self, topic: str, top_k: int = 10) -> list[dict]:
        matched = [(i, self.nodes[i]) for i in range(len(self.nodes))
                   if self.nodes[i].topic == topic]
        matched.sort(key=lambda x: x[1].timestamp, reverse=True)
        return [{"index": i, **n.to_dict(), "score": 1.0} for i, n in matched[:top_k]]

    def stats(self) -> dict:
        n = len(self.nodes)
        topics = sorted(set(n.topic for n in self.nodes if n.topic))
        edges = sum(len(v) for v in self.adj_list.values()) // 2 if self.adj_list else 0
        roles = {}
        for node in self.nodes:
            roles[node.role] = roles.get(node.role, 0) + 1
        avg_degree = 2 * edges / max(n, 1)
        density = edges / max(n * (n - 1) / 2, 1) * 100
        return {
            "total_memories": n,
            "topics": topics,
            "num_topics": len(topics),
            "edges": edges,
            "avg_degree": round(avg_degree, 2),
            "density": round(density, 2),
            "roles": roles,
            "storage_path": self.storage_path,
        }


# ══════════════════════════════════════════════════════════
# 交互式 CLI
# ══════════════════════════════════════════════════════════

def interactive_cli(mem: MemoryGraph):
    """交互式命令行界面"""
    print("\n" + "=" * 60)
    print("  MyGRAGMemory — 交互式记忆系统")
    print("=" * 60)
    print("  s <role> <content>          存储记忆")
    print("  conv <topic> <user> <asst>   存储对话")
    print("  q <query>                    GRAG检索")
    print("  topic <name>                 主题回忆")
    print("  topics                       列出主题")
    print("  stats                        统计")
    print("  exit                         退出")
    while True:
        try:
            cmd = input("mem> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not cmd or cmd == "exit":
            break
        parts = cmd.split(maxsplit=1)
        action = parts[0].lower()
        try:
            if action == "s" and len(parts) > 1:
                a = parts[1].split(maxsplit=1)
                if len(a) == 2:
                    nid = mem.store(a[0], a[1])
                    print(f"  Stored #{nid} ({a[0]})")
            elif action == "conv" and len(parts) > 1:
                a = parts[1].split(maxsplit=2)
                if len(a) == 3:
                    uid, aid = mem.store_conversation(a[1], a[2], topic=a[0])
                    print(f"  Stored #{uid},{aid} topic={a[0]}")
            elif action == "q" and len(parts) > 1:
                for r in mem.search(parts[1], top_k=5):
                    c = r["content"][:60]
                    print(f"  [{r['topic']}] {r['role']} s={r['score']:.4f} | {c}")
            elif action == "topic" and len(parts) > 1:
                for r in mem.recall(parts[1]):
                    print(f"  [{r['topic']}] {r['role']}: {r['content'][:60]}")
            elif action == "topics":
                topics = sorted(set(n.topic for n in mem.nodes))
                print(f"  Topics: {', '.join(topics) or '(none)'}")
            elif action == "stats":
                s = mem.stats()
                print(f"  Memories: {s['total_memories']}, Edges: {s['edges']}, "
                      f"Density: {s['density']}%")
            else:
                print(f"  Unknown: {action}")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MyGRAGMemory")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--cmd", type=str)
    parser.add_argument("--storage", type=str, default="results/memories.json")
    args = parser.parse_args()

    if args.reset and os.path.exists(args.storage):
        os.remove(args.storage)
        print("Memory reset.")
    else:
        mem = MemoryGraph({"storage_path": args.storage})
        if args.cmd:
            for r in mem.search(args.cmd.split(maxsplit=1)[-1] if " " in args.cmd else args.cmd):
                print(json.dumps(r, ensure_ascii=False))
        elif args.interactive:
            interactive_cli(mem)
        else:
            from experiments.run_experiments import run_all
            run_all()

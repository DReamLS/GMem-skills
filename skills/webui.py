#!/usr/bin/env python3
"""
GRAG Memory Graph WebUI — 知识图谱可视化前端

通过 HTTP 服务暴露记忆图结构，支持：
  - 力导向图可视化 (D3.js)
  - 节点/边详情查看
  - 图结构统计
  - GRAG检索实时测试

启动:
  python -m skills.webui --storage results/large_exp_memories.json
  # 访问 http://localhost:5000
"""
import argparse
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skills.memory_skill import MemoryGraph


# ══════════════════════════════════════════════════════════
# API 处理器
# ══════════════════════════════════════════════════════════

class GraphAPIHandler(BaseHTTPRequestHandler):
    """HTTP API + 静态文件服务"""

    mem: MemoryGraph = None  # 由外部注入

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        # API 路由
        if path == "/api/graph":
            self._send_json(self._get_graph_data())
        elif path == "/api/stats":
            self._send_json(GraphAPIHandler.mem.stats())
        elif path == "/api/search":
            q = params.get("q", [""])[0]
            top_k = int(params.get("top_k", ["10"])[0])
            results = GraphAPIHandler.mem.search(q, top_k=top_k)
            self._send_json({"query": q, "results": results})
        elif path == "/api/topics":
            topics = sorted(set(n.topic for n in GraphAPIHandler.mem.nodes))
            self._send_json({"topics": topics})
        elif path == "/api/node":
            nid = int(params.get("id", ["0"])[0])
            if 0 <= nid < len(GraphAPIHandler.mem.nodes):
                node = GraphAPIHandler.mem.nodes[nid]
                neighbors = list(GraphAPIHandler.mem.adj_list.get(nid, []))
                neighbor_data = [{
                    "id": GraphAPIHandler.mem.nodes[n].id,
                    "content": GraphAPIHandler.mem.nodes[n].content[:60],
                    "topic": GraphAPIHandler.mem.nodes[n].topic,
                    "role": GraphAPIHandler.mem.nodes[n].role,
                } for n in neighbors]
                self._send_json({
                    "node": node.to_dict(),
                    "degree": len(neighbors),
                    "neighbors": neighbor_data,
                })
            else:
                self._send_json({"error": "node not found"}, 404)
        elif path == "/" or path == "":
            self._serve_file("index.html", "text/html; charset=utf-8")
        else:
            self._send_json({"error": "not found"}, 404)

    def _get_graph_data(self) -> dict:
        """构建供 D3.js 使用的图数据"""
        mem = GraphAPIHandler.mem
        nodes = []
        for i, n in enumerate(mem.nodes):
            nodes.append({
                "id": i,
                "label": n.content[:50],
                "topic": n.topic,
                "role": n.role,
                "timestamp": n.timestamp,
                "size": len(mem.adj_list.get(i, [])) + 5,
            })
        edges = []
        seen = set()
        for i, adj in mem.adj_list.items():
            for j in adj:
                key = (min(i, j), max(i, j))
                if key not in seen:
                    seen.add(key)
                    # 确定边类型
                    n1, n2 = mem.nodes[i], mem.nodes[j]
                    if n1.topic == n2.topic:
                        edge_type = "topic"
                    elif abs(i - j) == 1 and n1.topic == n2.topic:
                        edge_type = "temporal"
                    else:
                        edge_type = "semantic"
                    edges.append({
                        "source": i,
                        "target": j,
                        "type": edge_type,
                    })
        # 太大时采样（超过500节点就只显示部分）
        MAX_VIS_NODES = 300
        if len(nodes) > MAX_VIS_NODES:
            # 保留枢纽节点 + 随机采样
            degree_sorted = sorted(range(len(nodes)),
                                   key=lambda i: len(mem.adj_list.get(i, [])),
                                   reverse=True)
            keep = set(degree_sorted[:MAX_VIS_NODES // 2])
            import random
            random.seed(42)
            keep.update(random.sample(degree_sorted[MAX_VIS_NODES // 2:],
                                       min(MAX_VIS_NODES // 2, len(nodes) - MAX_VIS_NODES // 2)))
            nodes = [n for i, n in enumerate(nodes) if i in keep]
            # 重建id映射
            id_map = {old: new for new, old in enumerate(sorted(keep))}
            for n, old_id in zip(nodes, sorted(keep)):
                n["id"] = id_map[old_id]
            edges = [e for e in edges if e["source"] in keep and e["target"] in keep]
            for e in edges:
                e["source"] = id_map[e["source"]]
                e["target"] = id_map[e["target"]]

        return {"nodes": nodes, "edges": edges}

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, filename, mime):
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(here, "templates", filename), "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self._send_json({"error": f"{filename} not found"}, 404)

    def log_message(self, format, *args):
        print(f"  [WEB] {args[0]} {args[1]} {args[2]}")


# ══════════════════════════════════════════════════════════
# HTML 模板（内嵌）
# ══════════════════════════════════════════════════════════

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>GRAG Memory Graph Visualizer</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, 'Microsoft YaHei', sans-serif; background: #1a1a2e; color: #eee; overflow: hidden; }
#header { position: fixed; top: 0; left: 0; right: 0; height: 48px; background: #16213e; display: flex; align-items: center; padding: 0 20px; z-index: 100; border-bottom: 1px solid #0f3460; }
#header h1 { font-size: 16px; color: #e94560; }
#header .stats { margin-left: 30px; font-size: 12px; color: #aaa; }
#header .stats span { margin-right: 15px; }
#sidebar { position: fixed; top: 48px; right: 0; width: 320px; bottom: 0; background: #16213e; padding: 15px; overflow-y: auto; z-index: 50; border-left: 1px solid #0f3460; }
#sidebar h3 { font-size: 14px; color: #e94560; margin-bottom: 10px; }
#sidebar .info { font-size: 12px; color: #ccc; margin-bottom: 5px; }
#sidebar .info b { color: #4fc3f7; }
#search-box { display: flex; margin-bottom: 15px; }
#search-box input { flex: 1; padding: 8px; border: 1px solid #0f3460; border-radius: 4px 0 0 4px; background: #1a1a2e; color: #eee; font-size: 13px; }
#search-box button { padding: 8px 15px; border: none; background: #e94560; color: white; border-radius: 0 4px 4px 0; cursor: pointer; font-size: 13px; }
#search-box button:hover { background: #c23152; }
#search-results { margin-top: 10px; }
.search-item { padding: 8px; margin-bottom: 5px; background: #1a1a2e; border-radius: 4px; border-left: 3px solid #4fc3f7; font-size: 12px; cursor: pointer; }
.search-item:hover { background: #0f3460; }
.search-item .tag { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10px; margin-right: 5px; color: white; }
.node-detail { padding: 10px; background: #1a1a2e; border-radius: 4px; margin-bottom: 10px; }
.node-detail .label { font-size: 13px; font-weight: bold; color: #eee; }
.node-detail .meta { font-size: 11px; color: #999; margin-top: 5px; }
.node-detail .neighbors { margin-top: 8px; font-size: 11px; }
.neighbor-item { padding: 3px 0; color: #aaa; }
#graph-container { position: fixed; top: 48px; left: 0; right: 320px; bottom: 0; }
.tooltip { position: absolute; padding: 8px 12px; background: rgba(22, 33, 62, 0.95); border: 1px solid #0f3460; border-radius: 4px; font-size: 12px; pointer-events: none; z-index: 200; max-width: 300px; }
.legend { position: fixed; bottom: 20px; left: 20px; background: rgba(22, 33, 62, 0.9); padding: 10px 15px; border-radius: 4px; font-size: 11px; z-index: 50; }
.legend-item { display: inline-block; margin-right: 15px; }
.legend-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }
.legend-line { display: inline-block; width: 20px; height: 2px; margin-right: 4px; vertical-align: middle; }
svg text { pointer-events: none; }
</style>
</head>
<body>

<div id="header">
  <h1>🕸️ GRAG Memory Graph</h1>
  <div class="stats">
    <span id="stat-nodes">Nodes: --</span>
    <span id="stat-edges">Edges: --</span>
    <span id="stat-density">Density: --</span>
    <span id="stat-topics">Topics: --</span>
  </div>
</div>

<div id="graph-container">
  <svg width="100%" height="100%"></svg>
</div>

<div class="legend">
  <span class="legend-item"><span class="legend-dot" style="background:#4fc3f7"></span>User</span>
  <span class="legend-item"><span class="legend-dot" style="background:#e94560"></span>Assistant</span>
  <span style="margin:0 10px;color:#555">|</span>
  <span class="legend-item"><span class="legend-line" style="background:#4fc3f7"></span>Semantic</span>
  <span class="legend-item"><span class="legend-line" style="background:#f1c40f"></span>Topic</span>
  <span class="legend-item"><span class="legend-line" style="background:#2ecc71"></span>Temporal</span>
</div>

<div id="sidebar">
  <h3>🔍 GRAG Search</h3>
  <div id="search-box">
    <input id="search-input" placeholder="输入查询..." onkeydown="if(event.key==='Enter') search()">
    <button onclick="search()">Search</button>
  </div>
  <div id="search-results"></div>
  <hr style="border-color:#0f3460;margin:15px 0">
  <h3>📋 Node Info</h3>
  <div id="node-info">点击节点查看详情</div>
</div>

<div id="tooltip" class="tooltip" style="display:none"></div>

<script>
const svg = d3.select("svg");
const width = () => document.getElementById("graph-container").clientWidth;
const height = () => document.getElementById("graph-container").clientHeight;
let graphData = { nodes: [], edges: [] };
let simulation;

const TOPIC_COLORS = {
  "rag": "#e94560", "python": "#4fc3f7", "deep-learning": "#ff9800",
  "database": "#9c27b0", "nlp": "#4caf50", "mlops": "#00bcd4",
  "algorithm": "#ff5722", "system": "#607d8b", "general": "#888"
};
const EDGE_COLORS = { "semantic": "#4fc3f7", "topic": "#f1c40f", "temporal": "#2ecc71" };

// 加载图数据
fetch("/api/graph")
  .then(r => r.json())
  .then(data => {
    graphData = data;
    document.getElementById("stat-nodes").textContent = `Nodes: ${data.nodes.length}`;
    document.getElementById("stat-edges").textContent = `Edges: ${data.edges.length}`;
    fetch("/api/stats").then(r=>r.json()).then(s => {
      document.getElementById("stat-density").textContent = `Density: ${s.density}%`;
      document.getElementById("stat-topics").textContent = `Topics: ${s.num_topics}`;
    });
    renderGraph(data);
  });

function renderGraph(data) {
  svg.selectAll("*").remove();
  const g = svg.append("g");
  const zoom = d3.zoom().scaleExtent([0.1, 8]).on("zoom", (e) => g.attr("transform", e.transform));
  svg.call(zoom);

  // Edges
  const link = g.append("g").selectAll("line")
    .data(data.edges).join("line")
    .attr("stroke", d => EDGE_COLORS[d.type] || "#555")
    .attr("stroke-width", 1)
    .attr("opacity", 0.4);

  // Nodes
  const node = g.append("g").selectAll("circle")
    .data(data.nodes).join("circle")
    .attr("r", d => Math.min(Math.max(d.size, 4), 20))
    .attr("fill", d => TOPIC_COLORS[d.topic] || "#888")
    .attr("opacity", 0.85)
    .attr("stroke", "#fff")
    .attr("stroke-width", 1)
    .call(d3.drag()
      .on("start", (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
      .on("end", (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; })
    )
    .on("mouseover", (e, d) => {
      const tip = document.getElementById("tooltip");
      tip.style.display = "block";
      tip.innerHTML = `<b>${d.label}</b><br>Topic: ${d.topic} | Role: ${d.role} | Degree: ${d.size - 5}`;
      tip.style.left = (e.pageX + 10) + "px";
      tip.style.top = (e.pageY - 10) + "px";
    })
    .on("mouseout", () => document.getElementById("tooltip").style.display = "none")
    .on("click", (e, d) => showNodeInfo(d.id));

  // Labels
  g.append("g").selectAll("text")
    .data(data.nodes).join("text")
    .text(d => d.label.length > 15 ? d.label.slice(0, 15) + ".." : d.label)
    .attr("font-size", 10)
    .attr("dx", d => Math.min(Math.max(d.size, 4), 20) + 3)
    .attr("dy", 3)
    .attr("fill", "#ccc")
    .attr("opacity", 0.7);

  // Simulation
  simulation = d3.forceSimulation(data.nodes)
    .force("link", d3.forceLink(data.edges).id(d => d.id).distance(80))
    .force("charge", d3.forceManyBody().strength(-150))
    .force("center", d3.forceCenter(width() / 2, height() / 2))
    .force("collision", d3.forceCollide().radius(d => Math.min(Math.max(d.size, 4), 20) + 5))
    .on("tick", () => {
      link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
          .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
      node.attr("cx", d => d.x).attr("cy", d => d.y);
      g.selectAll("text").attr("x", d => d.x).attr("y", d => d.y);
    });
}

// 节点详情
function showNodeInfo(id) {
  fetch(`/api/node?id=${id}`).then(r => r.json()).then(data => {
    const d = document.getElementById("node-info");
    let html = `<div class="node-detail">
      <div class="label">${data.node.content}</div>
      <div class="meta">Topic: ${data.node.topic} | Role: ${data.node.role} | Degree: ${data.degree}</div>`;
    if (data.neighbors.length > 0) {
      html += `<div class="neighbors"><b>Neighbors (${data.neighbors.length}):</b>`;
      data.neighbors.forEach(n => {
        html += `<div class="neighbor-item" onclick="showNodeInfo(${n.id})">↳ [${n.topic}] ${n.content}</div>`;
      });
      html += `</div>`;
    }
    html += `</div>`;
    d.innerHTML = html;
  });
}

// GRAG搜索
function search() {
  const q = document.getElementById("search-input").value.trim();
  if (!q) return;
  fetch(`/api/search?q=${encodeURIComponent(q)}&top_k=10`)
    .then(r => r.json()).then(data => {
      const div = document.getElementById("search-results");
      if (data.results.length === 0) {
        div.innerHTML = '<div class="info" style="color:#999">No results</div>';
        return;
      }
      let html = `<div class="info">Top ${data.results.length} results:</div>`;
      data.results.forEach((r, i) => {
        const color = TOPIC_COLORS[r.topic] || "#888";
        html += `<div class="search-item" onclick="showNodeInfo(${r.id})">
          <span class="tag" style="background:${color}">${r.topic}</span>
          <b>${r.role}</b> score=${r.score.toFixed(3)}
          <br>${r.content.slice(0, 60)}</div>`;
      });
      div.innerHTML = html;
    });
}

window.addEventListener("resize", () => {
  svg.attr("width", width()).attr("height", height());
  if (simulation) simulation.force("center", d3.forceCenter(width() / 2, height() / 2)).alpha(0.3).restart();
});
</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════
# 启动服务器
# ══════════════════════════════════════════════════════════

def write_html():
    """写入HTML模板文件"""
    here = os.path.dirname(os.path.abspath(__file__))
    tpl_dir = os.path.join(here, "templates")
    os.makedirs(tpl_dir, exist_ok=True)
    with open(os.path.join(tpl_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE)


def start_server(memory_path: str, port: int = 5000):
    """启动Web服务器"""
    write_html()

    mem = MemoryGraph({"storage_path": memory_path})
    stats = mem.stats()
    print(f"  [WEB] Loaded memory graph: {stats['total_memories']} nodes, {stats['edges']} edges")

    GraphAPIHandler.mem = mem

    server = HTTPServer(("0.0.0.0", port), GraphAPIHandler)
    print(f"  [WEB] Server started at http://localhost:{port}")
    print(f"  [WEB] Press Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  [WEB] Server stopped.")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRAG Memory Graph WebUI")
    parser.add_argument("--storage", default="results/large_exp_memories.json",
                        help="记忆图存储路径")
    parser.add_argument("--port", type=int, default=5000, help="端口号")
    args = parser.parse_args()

    start_server(args.storage, args.port)

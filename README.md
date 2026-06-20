# MyGRAGMemory: GRAG-Enhanced Agent Memory System

基于图检索增强生成（GRAG）的智能体记忆系统，可作为 Claude Code Skill 使用。

**GitHub**: [https://github.com/DReamLS/GMem-skills](https://github.com/DReamLS/GMem-skills)

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行完整实验
python -m experiments.run_experiments

# 交互模式
python -m skills.memory_skill --interactive
```

## 项目结构

```
myproject/
├── README.md                 # 本文件
├── requirements.txt          # Python依赖
├── skills/                   # 核心技能代码
│   ├── __init__.py
│   ├── memory_skill.py       # 记忆技能主模块
│   ├── config.py             # 配置管理
│   └── skill_template.yaml   # Claude Skill 模板
├── experiments/              # 实验代码
│   ├── __init__.py
│   ├── run_experiments.py    # 实验运行器
│   └── evaluate_results.py   # 评估与指标计算
├── docs/                     # 文档
│   ├── 实验方法.md            # 实验方法论
│   └── 评估过程.md            # 评估过程
├── results/                  # 实验结果输出
└── .claude/
    ├── settings.json
    └── skills/
        └── memory-grag.yaml  # Claude Skill 注册
```

## 作为 Claude Skill 使用

1. 将本项目放入 Claude Code 的 skills 目录
2. 运行 `/memory-grag` 加载技能
3. 技能将自动安装依赖并进入交互式记忆模式

或在 Claude Code 中直接调用：
```
/memory-grag 检索关于RAG技术的记忆
```

## 核心功能

| API | 功能 |
|-----|------|
| `store(role, content, topic)` | 存储单条记忆 |
| `store_conversation(usr, asst, topic)` | 存储完整对话 |
| `search(query, top_k)` | GRAG检索记忆 |
| `recall(topic, top_k)` | 按主题回忆 |
| `stats()` | 系统统计 |

## 实验结果

在5主题17轮对话（34条记忆）上验证：
- GRAG跨主题检索覆盖率：**100%**
- PPR收敛迭代次数：**7次**
- 图密度：**39.7%**
- 平均检索延迟：**<50ms**

# Vector Graph Pipeline (Query-only encode • One-shot similarity • Local APPNP • BFS • Rerank)

> 针对“**向量图已编码**、**仅在查询时使用嵌入模型**、**一次性算全量相似度并缓存**、**APPNP 仅在候选局部子图上运行**”的实现版本。

## 目录结构
```
vector_graph_pipeline/
  code/
    util/
      io_paths.py           # 路径与文件名约定（默认使用 D:\kg_out ...），输出文件“语义拆分”
      vector_io.py          # 元信息/向量加载、一次性相似度计算与缓存（只编码 query）
      seed_retrieval.py     # 基于相似度选择 Top-K 种子 chunk，构造 G0，并将边权写回
      bfs_utils.py          # ✅ 修复的 BFS（MultiGraph：4元组解包，稳健拿邻居）
      appnp_local.py        # 稀疏 APPNP（仅在 k-hop 诱导局部子图上跑），并支持边选择策略
      pipeline.py           # 端到端 CLI：召回 → 局部 APPNP → BFS → 重排（可独立运行）
  README.md
```

> **依赖**：`networkx`, `numpy`, `scipy`, `sentence-transformers`, `FlagEmbedding`。  
> **数据/模型不在此目录**，仍按你现有的 `D:\kg_out\...` 约定存放。

## 输出文件（语义拆分，避免覆盖）
- `D:\kg_out\seed_subgraph.json`
- `D:\kg_out\appnp_subgraph.json`
- `D:\kg_out\all_chunk_similarities.npy`
- `D:\kg_out\all_chunk_similarities.meta.json`
- `D:\kg_out\seed_chunk_scores.json`
- `D:\kg_out\appnp_node_scores.json`
- `D:\kg_out\vector_pipeline_result.json`

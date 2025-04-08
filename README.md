# --计算机八股问答系统
遇到问题
问题1.构建FAISS索引使用的是 sentence_transformers 的 all-MiniLM-L6-v2 模型，而现在查询时使用的是 OpenAI的text-embedding-ada-002模型。这两个模型生成的向量维度不一致：
all-MiniLM-L6-v2 的维度是：384
OpenAI text-embedding-ada-002 的维度是：1536(已解决)
问题2.模型输出的答案来源不是我在知识库中提供的，想到的解决方法1.构建新知识库，使用没有图片，结构化更好的文档 2.微调模型

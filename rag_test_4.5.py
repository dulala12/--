import openai
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st

class EightLeggedSystem:
    def __init__(self, model_name='all-MiniLM-L6-v2', openai_api_key='YOUR-API-KEY'):
        self.model = SentenceTransformer(model_name)
        openai.api_key = openai_api_key

        with open('metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)

        with open('chunks.pkl', 'rb') as f:
            self.chunks = pickle.load(f)

        self.vector_index = faiss.read_index('vector_index.faiss')

    def get_embedding(self, text):
        embedding = self.model.encode([text])[0]
        return np.array(embedding).astype('float32')

    def query_knowledge_base(self, question, top_k=3):
        query_vector = self.get_embedding(question)
        _, indices = self.vector_index.search(np.array([query_vector]), top_k)
        return [(self.metadata[idx], self.chunks[idx]) for idx in indices[0]]

    def generate_answer(self, question, top_k=3):
        try:
            context_docs = self.query_knowledge_base(question, top_k)
            context_str = "\n\n".join([doc[1] for doc in context_docs])

            prompt = f"""
角色定位：

你是一名知识扎实、实践经验丰富、表达流畅的优秀计算机专业学生，具备扎实的理论基础和丰富的工程实践经验。

面试要求：

逻辑清晰、思路严谨：

每个问题回答时，请先简要概述你的结论，再详细阐述推理过程。

确保论据充分，论证步骤清晰，并尽量用实例辅助说明。

在讨论复杂问题时，可使用分点说明、流程图或伪代码，使答案更具层次性。

专业性和实战经验：

针对技术性问题（如算法、数据结构、操作系统、计算机网络等）时，结合相应的理论知识和实际项目经验进行说明。

遇到设计题目时，先给出高层次的设计思路，再逐步展开细节描述，并讨论可能的改进方案和风险点。

语言简洁、表达准确：

使用专业术语和概念时，请确保解释清楚，避免过于抽象。

针对面试官可能存在的疑问，提前预判并进行解释，展示你的全面性和深度思考。

问题应对策略：

对不太熟悉的问题，可尝试从基本概念出发，逐步推导到复杂问题，体现你的学习能力和解决问题的逻辑。

保持自信和耐心，适时举例或类比，使复杂概念更容易理解。

【相关资料】
{context_str}

【问题】
{question}

【计算机面试格式答案】
"""

            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.1,
            )

            return {"answer": response.choices[0].message.content.strip(), "contexts": [doc[1] for doc in context_docs]}

        except Exception as e:
            return {"error": str(e)}

# Streamlit UI
st.title("计算机面试八股文问答系统")

question = st.text_input("请输入你的问题：")

if question:
    system = EightLeggedSystem()
    result = system.generate_answer(question)

    if "error" in result:
        st.error(f"错误：{result['error']}")
    else:
        st.subheader("八股文答案")
        st.write(result["answer"])

        st.subheader("参考来源")
        for i, ctx in enumerate(result["contexts"], 1):
            st.write(f"来源{i}: {ctx[:200]}...")
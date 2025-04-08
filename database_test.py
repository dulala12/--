import re
import numpy as np
from PyPDF2 import PdfReader  # 用于 PDF 文本提取
from sentence_transformers import SentenceTransformer
import faiss
import pickle  # 用于保存元数据

# --- Step 1: PDF 文本提取 ---
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        # 注意：某些 PDF 的 extract_text() 方法效果不佳，可能需要其它库试试
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# 读取书的文本
pdf_path1 = r"C:\Users\Administrator\Desktop\demo\book.pdf"  # 替换为你的 PDF 文件路径

book1_text = extract_text_from_pdf(pdf_path1)


# --- Step 2: 文本预处理与切分 ---
def split_text(text, max_length=500):
    """
    根据 max_length（字符数）将文本按句子分块
    可根据实际情况调整：如果希望按语义或固定 token 数切分，
    可以使用 nltk 等工具包进一步处理。
    """
    # 先简单按照换行符拆分，再逐步合并成长度不超过 max_length 的块
    sentences = re.split(r'\n+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        # 如果当前块长度加上新句子超过 max_length，则保存当前块，并重置
        if len(current_chunk) + len(sentence) > max_length:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
        else:
            current_chunk += sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

chunks1 = split_text(book1_text, max_length=500)


# --- Step 3: 构建文本块列表与元数据 ---
chunks = []      # 存储所有文本块
metadata = []    # 存储每个文本块的元数据（例如所属书籍、块索引等）

for idx, chunk in enumerate(chunks1):
    chunks.append(chunk)
    metadata.append({"book": pdf_path1, "chunk_index": idx})


print(f"共提取 {len(chunks)} 个文本块.")

# --- Step 4: 生成嵌入向量 ---
# 使用预训练的 SentenceTransformer 模型生成向量嵌入
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)
embeddings = model.encode(chunks, show_progress_bar=True)

# 确保 embeddings 为 float32 格式（FAISS 要求）
embeddings = np.array(embeddings).astype('float32')

# --- Step 5: 利用 FAISS 构建向量数据库 ---
# 这里以 L2 距离为例，可根据需求选择内积或其它距离度量
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("向量数据库构建完成！")
print(f"数据库中包含 {index.ntotal} 个向量.")

# --- Step 6: 持久化存储（可选） ---
# 如果希望将构建好的向量数据库保存到磁盘，可以用 faiss.write_index
faiss.write_index(index, "vector_index.faiss")
# 同时，将元数据信息保存，比如用 pickle 存储
with open("metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)
print("向量数据库和元数据已保存。")

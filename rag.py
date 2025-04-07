from llama_index.core import SimpleDirectoryReader, Document, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

print("DEBUG imports rolaram!")

DOCUMENT_PATH = "/home/pfpimenta/2025_04_06_rag/1706.03762v7.pdf"

print("DEBUG lendo documentos...")
documents = SimpleDirectoryReader(input_files=[DOCUMENT_PATH]).load_data()
document = Document(text="\n\n".join([doc.text for doc in documents]))

print("DEBUG carregando LLM...")
Settings.llm = Ollama(model="llama2", request_timeout=200.0)
print("DEBUG carregando embedding model...")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

print("DEBUG inicializando VectorStoreIndex...")
index = VectorStoreIndex.from_documents([document])

# a document summary index needs both an llm and embed model
# for the constructor
print("DEBUG inicializando query_engine...")
query_engine = index.as_query_engine()

question = "What is the attention mechanism?"
print(f"DEBUG perguntando: \'{question}\' ...")
response = query_engine.query(question)
print(f"RAG answer: \n\n{str(response)}")
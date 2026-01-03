import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "research-rag-llama3"

# 1. Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists, if not create it
existing_indexes = [index.name for index in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    print(f"Creating index: {INDEX_NAME}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384, # IMPORTANT: HuggingFace MiniLM uses 384 dimensions (OpenAI uses 1536)
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# 2. Setup Embedding Model (Local & Free)
print("Loading Embedding Model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Load PDFs
print("Loading PDFs from 'data/' folder...")
documents = []
data_folder = "data"
for file in os.listdir(data_folder):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(data_folder, file)
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        print(f" - Loaded {file}: {len(docs)} pages")
        documents.extend(docs)

# 4. Split Text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)}")

# 5. Store in Pinecone
print("Uploading vectors to Pinecone (this might take a minute)...")
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=INDEX_NAME
)

print("✅ Success! Database is ready.")
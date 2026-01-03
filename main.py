import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA

# Load keys
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "research-rag-llama3"

# 1. Setup Llama 3 via OpenRouter
# We use ChatOpenAI class but point it to OpenRouter URL
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="meta-llama/llama-3-8b-instruct",
    temperature=0
)

# 2. Setup the Retriever (Connect to Pinecone)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME, 
    embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks

# 3. Create the Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 4. The Loop (Chat Interface)
print("🤖 Research Assistant Ready! (Type 'exit' to quit)")
print("-" * 50)

while True:
    query = input("\nAsk a question: ")
    if query.lower() == "exit":
        break
    
    # Run the query
    print("Thinking...")
    response = qa_chain.invoke({"query": query})
    
    # Print Answer
    print(f"\nAnswer:\n{response['result']}")
    
    # Print Sources (Professional Touch)
    print("\n[Sources Used]:")
    for doc in response['source_documents']:
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'Unknown')
        print(f"- {os.path.basename(source)} (Page {page})")
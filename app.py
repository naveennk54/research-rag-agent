import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA

# Page Configuration
st.set_page_config(page_title="BioGen Research Assistant", layout="wide")

# Load Keys
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "research-rag-llama3"

# --- Sidebar ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3022/3022340.png", width=100)
st.sidebar.title("📚 Research Docs")
st.sidebar.markdown("This tool searches internal research papers to answer your questions.")
st.sidebar.header("Current Database:")
st.sidebar.success("✅ Connected to Pinecone")
st.sidebar.info(f"Index: {INDEX_NAME}")

# --- Main Content ---
st.title("🧬 Advanced RAG Pipeline")
st.markdown("Ask questions about **LLMs in Healthcare**, **GatorTron**, or **Medical Ethics**.")

# Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to get answer
def get_answer(query):
    # 1. Connect to LLM
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        model="meta-llama/llama-3-8b-instruct",
        temperature=0
    )
    
    # 2. Connect to Pinecone
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME, 
        embedding=embeddings
    )
    
    # 3. Create Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    return qa_chain.invoke({"query": query})

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input Area
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Answer
    with st.chat_message("assistant"):
        with st.spinner("Analyzing research papers..."):
            response = get_answer(prompt)
            answer_text = response['result']
            
            # Format Sources
            sources_text = "\n\n**Sources:**"
            unique_sources = set()
            for doc in response['source_documents']:
                source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                page_num = int(doc.metadata.get('page', 0)) + 1
                unique_sources.add(f"{source_name} (Page {page_num})")
            
            for source in unique_sources:
                sources_text += f"\n- {source}"
            
            full_response = answer_text + sources_text
            st.markdown(full_response)
            
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
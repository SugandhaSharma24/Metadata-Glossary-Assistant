import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings

# ✅ Community imports
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.utilities import SQLDatabase

# === STREAMLIT UI ===
st.set_page_config(page_title="Healthcare RAG Assistant", page_icon="🩺")
st.title("🩺 Healthcare RAG Assistant")

# ✅ STEP 1: API Key Input
openai_api_key = st.text_input("🔐 Enter your OpenAI API Key:", type="password")

# Don't run anything unless key is provided
if not openai_api_key:
    st.warning("Please enter your OpenAI API key to start.")
    st.stop()

# ✅ STEP 2: Set the key
os.environ["OPENAI_API_KEY"] = openai_api_key


# === SQL AGENT SETUP ===
@st.cache_resource
def load_sql_agent():
    db = SQLDatabase.from_uri("sqlite:///healthcare.db")
    llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=openai_api_key)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return create_sql_agent(llm=llm, toolkit=toolkit, verbose=False)


# === VECTOR RETRIEVER SETUP ===
@st.cache_resource
def load_vector_qa():
    loader = TextLoader("doctor_notes.txt")
    documents = loader.load()
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectordb = FAISS.from_documents(documents, embeddings)
    return RetrievalQA.from_chain_type(llm=ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key), retriever=vectordb.as_retriever())


# === LOGIC TO DECIDE WHERE TO ROUTE ===
def hybrid_agent(question, sql_agent, qa_chain):
    keywords = ["patient", "visit", "lab", "age", "result", "appointment", "glucose"]
    if any(kw in question.lower() for kw in keywords):
        return sql_agent.run(question)
    else:
        return qa_chain.run(question)


# === APP MAIN ===
st.write("Ask questions about structured data (patients, labs) or doctor notes.")
sql_agent = load_sql_agent()
qa_chain = load_vector_qa()

user_input = st.text_input("💬 Ask your question:")

if user_input:
    with st.spinner("Thinking..."):
        response = hybrid_agent(user_input, sql_agent, qa_chain)
    st.success("🧠 Answer:")
    st.write(response)

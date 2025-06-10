import os
import streamlit as st

# --- Streamlit page config ---
st.set_page_config(page_title="Healthcare RAG Assistant", page_icon="🩺", layout="wide")

# Sidebar: API key input and instructions
st.sidebar.header("🔐 API & Data Upload")
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
if not openai_api_key:
    st.sidebar.warning("API key required to proceed.")
    st.stop()

# Set environment variable EARLY so LangChain/OpenAI picks it up
os.environ["OPENAI_API_KEY"] = openai_api_key

# Now import LangChain and other dependent modules
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.utilities import SQLDatabase

from glossary_module import (
    setup_glossary_table,
    preload_glossary_terms,
    fetch_glossary_definition,
    generate_definition_from_context,
    validate_definition_with_context,
    score_definition_quality,
)

# --- Glossary Setup ---
setup_glossary_table()

glossary_terms = [
    ["Blood Pressure", "Blood Pressure is the pressure of circulating blood on the walls of blood vessels, measured in mmHg (millimeters of mercury)."],
    ["Heart Rate", "Heart Rate is number of heartbeats per minute (bpm), indicating cardiovascular activity."],
    ["Glucose", "Glucose is a simple sugar that is an important energy source in living organisms and a component of many carbohydrates."],
    ["BMI", "BMI full form is Body Mass Index. It is  a value derived from the mass and height of an individual, used to categorize underweight, normal, overweight, or obese."],
    ["Cholesterol", "Cholesterol is a type of fat found in your blood, essential for building healthy cells but high levels can increase heart disease risk."],
    ["Diagnosis", "Diagnosis is a process of identifying a disease, condition, or injury from its signs and symptoms."],
    ["Patient ID", "Patient ID unique identifier assigned to each patient in a healthcare system for tracking and record-keeping."],
    ["Clinical Trial", "Clinical Trial research study that tests how well new medical approaches work in people."],
    ["Medical History", "Medical History record of a patient's past illnesses, treatments, surgeries, and family health conditions."],
    ["Allergy", "Allergy is  condition in which the immune system reacts abnormally to a foreign substance."]
]

preload_glossary_terms(glossary_terms)

# --- Load SQL Agent with caching ---
@st.cache_resource
def load_sql_agent(api_key: str):
    db = SQLDatabase.from_uri("sqlite:///healthcare.db")
    llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=api_key)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return create_sql_agent(llm=llm, toolkit=toolkit, verbose=False)

# --- Load Vector QA with caching ---
@st.cache_resource
def load_vector_qa(api_key: str):
    loader = TextLoader("doctor_notes.txt")
    documents = loader.load()
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = FAISS.from_documents(documents, embeddings)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4", openai_api_key=api_key),
        retriever=retriever
    )
    return qa_chain

# Instantiate agents with the provided API key
sql_agent = load_sql_agent(openai_api_key)
qa_chain = load_vector_qa(openai_api_key)

# Hybrid agent logic (unchanged)
def hybrid_agent(question, sql_agent, qa_chain):
    glossary_terms_list = [term for term, _ in glossary_terms]
    question_lower = question.lower()
    glossary_question_prefixes = ["what does", "define", "meaning of", "explain", "what is", "tell me about"]
    is_glossary_question = any(prefix in question_lower for prefix in glossary_question_prefixes)
    matched_term = None

    for term in glossary_terms_list:
        if term.lower() in question_lower:
            matched_term = term
            break

    if is_glossary_question and matched_term:
        context_text = qa_chain.run(f"Provide context and usage examples of '{matched_term}' in medical notes.")
        context_sentences = context_text.split(". ")

        definition = fetch_glossary_definition(matched_term)
        if definition:
            validation_feedback = validate_definition_with_context(matched_term, definition, context_sentences)
            score = score_definition_quality(definition, context_sentences)
            generated_definition = generate_definition_from_context(context_sentences)

            response = f"""
**Term:** {matched_term}

**Existing Definition:** {definition}

🔍 **Validation Feedback:** {validation_feedback}

📊 **Quality Score:** {score}/10

📚 **Context from Metadata:**
{context_text}

📝 **Generated Draft Definition (from context):**
{generated_definition}
"""
            if score < 6:
                response += "\n⚠️ The current definition has a low quality score and should be reviewed."

            return response
        else:
            generated_definition = generate_definition_from_context(context_sentences)
            validation_feedback = validate_definition_with_context(matched_term, generated_definition, context_sentences)
            score = score_definition_quality(generated_definition, context_sentences)

            response = f"""**Term:** {matched_term} (Not found in glossary)

📚 Suggested Definition (based on context):
{generated_definition}

🔍 Validation Feedback: {validation_feedback}

📊 Quality Score: {score}/10

⚠️ Please review and consider adding this definition.
"""
            return response

    keywords = ["patient", "visit", "lab", "age", "result", "appointment", "glucose"]
    if any(kw in question_lower for kw in keywords):
        return sql_agent.run(question)

    return qa_chain.run(question)

# --- Streamlit UI ---
st.title("🩺 Healthcare RAG Assistant")
st.write("Ask questions about structured data (patients, labs), doctor notes, or glossary terms.")

user_input = st.text_input("💬 Ask your question:")

if user_input:
    with st.spinner("Thinking..."):
        answer = hybrid_agent(user_input, sql_agent, qa_chain)
    st.markdown("### 🧠 Answer:")
    st.markdown(answer)

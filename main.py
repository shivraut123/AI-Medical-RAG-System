import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
# ... other imports ...
from dotenv import load_dotenv  

# Load environment variables
load_dotenv()

# Securely fetch the key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("API Key not found. Please set it in the .env file.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Page Setup
st.set_page_config(page_title="MediBot Pro", page_icon="🩺", layout="wide")

# --- LOAD MODELS ---
@st.cache_resource
def load_llm():
    # Using the Gemini 2.5 Flash model
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    return llm

@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local("vectorstore/db_faiss", embeddings, allow_dangerous_deserialization=True)
    return db

# --- DYNAMIC PROMPTS ---
def get_prompt(mode):
    if mode == "Symptom Checker":
        template = """Act as a professional medical diagnostician. 
        Analyze the symptoms described in the user's question using the context provided.
        List potential conditions that match the symptoms.
        
        Context: {context}
        Question: {question}
        
        Format the answer as a bulleted list.
        Always end with: "Please consult a doctor for a definitive diagnosis."
        """
    elif mode == "First Aid Guide":
        template = """Act as an emergency first aid responder.
        Provide clear, step-by-step instructions on how to handle the situation described.
        Focus on immediate actions to stabilize the patient.
        
        Context: {context}
        Question: {question}
        
        Format the answer as numbered steps (Step 1, Step 2, etc.).
        If the situation is life-threatening, start with "CALL EMERGENCY SERVICES IMMEDIATELY."
        """
    else: # General Medical Consultant
        template = """Use the following pieces of medical context to answer the user's question.
        If you don't know the answer, just say that you don't know.
        
        Context: {context}
        Question: {question}
        
        Helpful Answer:
        """
    
    return PromptTemplate(template=template, input_variables=['context', 'question'])

# --- LOGIC ---
def get_response(query, db, llm, mode):
    prompt = get_prompt(mode)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    
    response = qa_chain.invoke({"query": query})
    return response

# --- FRONTEND (UI) ---
st.title("🩺 MediBot: Advanced Medical System")
st.markdown("### B.Tech Major Project - AI Healthcare Assistant")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Mode Selector
    mode = st.radio(
        "Select Assistant Mode:",
        ["Medical Consultant", "Symptom Checker", "First Aid Guide"]
    )
    
    st.markdown("---")
    
    # Hospital Finder Button
    if st.button("🏥 Find Nearby Hospitals"):
        st.markdown("[Click here to open Google Maps](https://www.google.com/maps/search/hospitals+near+me)")

    st.markdown("---")
    st.info(f"**Current Mode:** {mode}")
    
    # --- RAG INFO SECTION (The part you asked for) ---
    st.markdown("---")
    st.success("ℹ️ **Project Architecture**")
    st.markdown("""
    This system uses **RAG (Retrieval-Augmented Generation)**. 
    
    It does not "make up" answers. Instead, it:
    1. **Searches** the *Gale Encyclopedia of Medicine*.
    2. **Retrieves** relevant pages.
    3. **Generates** an answer based *only* on that medical data.
    """)
    st.warning("⚠️ **Disclaimer:** This is an AI project. Do not use for real emergencies.")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Running {mode} protocol..."):
            try:
                llm = load_llm()
                db = load_vector_store()
                
                result = get_response(prompt, db, llm, mode)
                answer = result['result']
                sources = result['source_documents']
                
                # Display Text Answer Only
                st.markdown(answer)
                
                # Source Evidence
                with st.expander("📄 View Medical Source"):
                    st.info(f"Source: Page {sources[0].metadata['page']}")
                    st.caption(sources[0].page_content[:400] + "...")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
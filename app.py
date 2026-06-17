import streamlit as st
import os
# from dotenv import load_dotenv

from utils import load_pdf, split_text, create_vector_store

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Load env
# load_dotenv()
# api_key = os.getenv("GOOGLE_API_KEY")
api_key = st.secrets["GEMINI_API_KEY"]

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    google_api_key=api_key
)

st.title("RAG App (Streamlit + Gemini)")

uploaded_file = st.file_uploader("Upload PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        text = load_pdf(uploaded_file)
    else: text = uploaded_file.read().decode("utf-8")

    st.success("File loaded successfully!")
    chunks = split_text(text)

    vectorstore = create_vector_store(chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""
    You are a highly accurate assistant.

    Your task is to answer the question using ONLY the context provided.

    Rules:
    - Use all relevant information from the context
    - Do NOT skip important points
    - If multiple points exist, list them clearly
    - If the context is partial, still give the most complete answer possible
    - Do not repeat irrelevant text

    Context:
    {context}

    Question:
    {input}

    Give a COMPLETE and well-structured answer:
    """)

    # RAG chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    query = st.text_input("Ask a question from your document:")
    if query:
        response = rag_chain.invoke({"input": query})
        st.write("### Answer:")
        st.write(response["answer"])
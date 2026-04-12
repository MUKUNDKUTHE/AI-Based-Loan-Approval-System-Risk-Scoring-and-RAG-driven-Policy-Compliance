import os
from unittest import result
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq 
import joblib

@st.cache_resource
def load_ml_model():
    return joblib.load("loan_pipeline.pkl")

from secret_api_keys import groq_api_key
os.environ['GROQ_API_KEY'] = groq_api_key

BASE_DB_PATH = "vectorstore"

PDF_PATHS = {
    "HDFC": "HDFC.pdf",
    "ICICI": "ICICI.pdf",
    "SBI": "SBI.pdf",
    "Union Bank": "UNIONBANK.pdf",
    "Kotak Mahindra Bank": "KOTAKMAHINDRABANK.pdf"
}

def load_vectorstores():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstores = {}

    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)

    for name, path in PDF_PATHS.items():
        db_path = os.path.join(BASE_DB_PATH, name)

        index_file = os.path.join(db_path, "index.faiss")
        pkl_file = os.path.join(db_path, "index.pkl")

        if os.path.exists(index_file) and os.path.exists(pkl_file):
            vectorstores[name] = FAISS.load_local(
                db_path, embeddings, allow_dangerous_deserialization=True
            )
        else:
            pdf_reader = PdfReader(path)
            docs = []

            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    chunks = splitter.split_text(text)
                    docs.extend(chunks)

            vs = FAISS.from_texts(docs, embeddings)
            vs.save_local(db_path)

            vectorstores[name] = vs

    return vectorstores

def create_vectorstore_from_uploaded_pdf(uploaded_file):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)

    docs = []
    pdf_reader = PdfReader(uploaded_file)

    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            chunks = splitter.split_text(text)
            docs.extend(chunks)

    vectorstore = FAISS.from_texts(docs, embeddings)
    return vectorstore

def get_vectorstore(bank_name):
    vectorstores = load_vectorstores()
    return vectorstores.get(bank_name)

def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",  
        temperature=0.5,
    )

def context_history(chat):
    history = chat[-3:]
    context = ""
    for i in history:
        context += f"User: {i['question']}\nAssistant: {i['answer']}\n"
    return context

def main():
    st.set_page_config(page_title="Loan Assistant", layout="wide")

    if "page" not in st.session_state:
        st.session_state.page = "form"

    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {"Chat 1": []}

    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "Chat 1"

    if "ml_prediction" not in st.session_state:
        st.session_state.ml_prediction = None

    if st.session_state.page == "form":

        st.title("Loan Assessment Form")

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", 18, 100)
            loan_amount = st.number_input("Loan Amount")
            months_employed = st.number_input("Months Employed")
            interest_rate = st.number_input("Interest Rate")
            dti = st.number_input("DTI Ratio")
            education = st.selectbox("Education", ["High School","Bachelor's","Master's"])
            marital = st.selectbox("Marital Status", ["Single","Married","Divorced"])
            dependents = st.selectbox("Has Dependents", ["Yes","No"])
            cosigner = st.selectbox("Has Co-Signer", ["Yes","No"])

        with col2:
            income = st.number_input("Income")
            credit_score = st.number_input("Credit Score")
            num_credit_lines = st.number_input("Number of Credit Lines")
            loan_term = st.number_input("Loan Term")
            employment = st.selectbox("Employment Type", ["Full-time","Part-time","Unemployed"])
            mortgage = st.selectbox("Has Mortgage", ["Yes","No"])
            purpose = st.selectbox("Loan Purpose", ["Auto","Business","Education","Other"])

        model = load_ml_model()

        if st.button("Submit & Calculate Risk"):
            data = pd.DataFrame([{
                "Age": age,
                "Income": income,
                "LoanAmount": loan_amount,
                "CreditScore": credit_score,
                "MonthsEmployed": months_employed,
                "NumCreditLines": num_credit_lines,
                "InterestRate": interest_rate,
                "LoanTerm": loan_term,
                "DTIRatio": dti,
                "Education": education,
                "EmploymentType": employment,
                "MaritalStatus": marital,
                "HasMortgage": mortgage,
                "HasDependents": dependents,
                "LoanPurpose": purpose,
                "HasCoSigner": cosigner
            }])

            st.session_state.user_data = data

            pd_value = model.predict_proba(data)[0][1]
            score = int(900 - (pd_value * 600))

            st.session_state.ml_prediction = {
                "label": f"Risk Score: {score}",
                "confidence": pd_value
            }

            st.success(f"Risk Score: {score}")

            st.session_state.page = "chat"
            st.rerun()

    elif st.session_state.page == "chat":

        st.title("Loan Assistant Chat")

        st.sidebar.title("Chats")

        chat_names = list(st.session_state.chat_sessions.keys())
        selected_chat = st.sidebar.selectbox("Select Chat", chat_names)

        st.session_state.current_chat = selected_chat
        
        active_chat = st.session_state.chat_sessions[st.session_state.current_chat]

        for msg in active_chat:
            with st.chat_message("user"):
                st.markdown(msg["question"])
            with st.chat_message("assistant"):
                st.markdown(msg["answer"])

        if st.sidebar.button(" New Chat"):
            new_name = f"Chat {len(chat_names)+1}"
            st.session_state.chat_sessions[new_name] = []
            st.session_state.current_chat = new_name
            st.rerun()

        bank = st.sidebar.selectbox(
            "Select Bank",
            ["HDFC", "ICICI", "SBI", "Union Bank", "Kotak Mahindra Bank"]
        )

        uploaded_file = st.sidebar.file_uploader("Upload Custom PDF", type="pdf")

        if st.sidebar.button("Back to Form"):
            st.session_state.page = "form"
            st.rerun()

        if uploaded_file is not None:
            vector_store = create_vectorstore_from_uploaded_pdf(uploaded_file)
        else:
            vector_store = get_vectorstore(bank)

        if uploaded_file:
            st.sidebar.success("Using uploaded PDF")
        else:
            st.sidebar.info(f"Using {bank} policy")
        llm = load_llm()

        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a loan approval assistant.
Based on the applicant details assist the user based on the context.If the question is not related to the context, say you don't know.
Answer directly to the question don't mention applicant details there.If the context is not regarding the loan approval then reply with upload a correct pdf.
Context:
{context}

Question:
{question}

Answer:"""
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True
        )

        question = st.chat_input("Ask about your loan...")

        if question:
            with st.chat_message("user"):
                st.markdown(question)

            previous_context = context_history(active_chat)

            if st.session_state.ml_prediction is not None:
                predicted_label = st.session_state.ml_prediction["label"]
                confidence = st.session_state.ml_prediction["confidence"]

                final_query = f"""
Previous Conversation:
{previous_context}
Applicant Information: {st.session_state.get("user_data")}
Loan Risk: {predicted_label}
Confidence: {confidence:.2f}

User Question:
{question}
"""
            else:
                final_query = question

            with st.spinner("Analyzing..."):
                result = qa_chain.invoke({"query": final_query})
                st.sidebar.markdown("Source Documents:")
                st.sidebar.write(result["source_documents"])
                answer = result["result"]

            with st.chat_message("assistant"):
                st.markdown(answer)

            active_chat.append({
                "question": question,
                "answer": answer
            })
if __name__ == "__main__":
    main()

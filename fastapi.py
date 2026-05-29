import os
import joblib
import pandas as pd
import pdfplumber
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

from langchain_groq import ChatGroq

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
import os
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = groq_api_key

BASE_DB_PATH = "vectorstore"

PDF_PATHS = {
    "HDFC": "HDFC.pdf",
    "ICICI": "ICICI.pdf",
    "SBI": "SBI.pdf",
    "Union Bank": "UNIONBANK.pdf",
    "Kotak Mahindra Bank": "KOTAKMAHINDRABANK.pdf"
}


model = joblib.load("loan_pipeline.pkl")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=50
)

vectorstores = {}

def load_vectorstores():

    global vectorstores

    for name, path in PDF_PATHS.items():

        db_path = os.path.join(BASE_DB_PATH, name)

        index_file = os.path.join(db_path, "index.faiss")
        pkl_file = os.path.join(db_path, "index.pkl")

        if os.path.exists(index_file) and os.path.exists(pkl_file):

            vectorstores[name] = FAISS.load_local(
                db_path,
                embeddings,
                allow_dangerous_deserialization=True
            )

        else:

            docs = []

            with pdfplumber.open(path) as pdf:

                for page in pdf.pages:

                    text = page.extract_text()

                    if text:

                        chunks = splitter.split_text(text)

                        docs.extend(chunks)

            vs = FAISS.from_texts(docs, embeddings)

            vs.save_local(db_path)

            vectorstores[name] = vs

load_vectorstores()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5
)


class LoanRequest(BaseModel):

    Age: int
    Income: float
    LoanAmount: float
    CreditScore: float
    MonthsEmployed: float
    NumCreditLines: float
    InterestRate: float
    LoanTerm: float
    DTIRatio: float

    Education: str
    EmploymentType: str
    MaritalStatus: str
    HasMortgage: str
    HasDependents: str
    LoanPurpose: str
    HasCoSigner: str

    bank: str
    question: str


@app.get("/")
def home():

    return {
        "message": "AI Loan Assistant Running"
    }

@app.post("/loan-assessment")

def loan_assessment(data: LoanRequest):


    df = pd.DataFrame([{

        "Age": data.Age,
        "Income": data.Income,
        "LoanAmount": data.LoanAmount,
        "CreditScore": data.CreditScore,
        "MonthsEmployed": data.MonthsEmployed,
        "NumCreditLines": data.NumCreditLines,
        "InterestRate": data.InterestRate,
        "LoanTerm": data.LoanTerm,
        "DTIRatio": data.DTIRatio,
        "Education": data.Education,
        "EmploymentType": data.EmploymentType,
        "MaritalStatus": data.MaritalStatus,
        "HasMortgage": data.HasMortgage,
        "HasDependents": data.HasDependents,
        "LoanPurpose": data.LoanPurpose,
        "HasCoSigner": data.HasCoSigner
    }])

    pd_value = model.predict_proba(df)[0][1]

    risk_score = int(900 - (pd_value * 600))



    vector_store = vectorstores.get(data.bank)


    prompt_template = PromptTemplate(

        input_variables=["context", "question"],

        template="""
You are a loan approval assistant.

Answer using the context.

If question is not related to the context say:
"I don't know the answer to that question based on the provided information."

Context:
{context}

Question:
{question}

Answer:
"""
    )


    qa_chain = RetrievalQA.from_chain_type(

        llm=llm,

        retriever=vector_store.as_retriever(
            search_kwargs={"k": 3}
        ),

        chain_type="stuff",

        chain_type_kwargs={
            "prompt": prompt_template
        },

        return_source_documents=True
    )



    final_query = f"""

Applicant Information:
{df.to_dict()}

Risk Score:
{risk_score}

User Question:
{data.question}
"""

    result = qa_chain.invoke({
        "query": final_query
    })


    return {

        "risk_score": risk_score,

        "confidence": float(pd_value),

        "answer": result["result"],

        "sources": [
            doc.page_content[:300]
            for doc in result["source_documents"]
        ]
    }


@app.post("/upload-pdf")

async def upload_pdf(
    file: UploadFile = File(...)
):

    docs = []

    with pdfplumber.open(file.file) as pdf:

        for page in pdf.pages:

            text = page.extract_text()

            if text:

                chunks = splitter.split_text(text)

                docs.extend(chunks)

    vs = FAISS.from_texts(docs, embeddings)

    vectorstores["CUSTOM"] = vs

    return {
        "message": "Custom PDF uploaded successfully"
    }

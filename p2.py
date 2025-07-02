

import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file if available
load_dotenv(find_dotenv())

# Step 1: Setup LLM (Mistral with HuggingFace Inference API)
HF_TOKEN = os.environ.get("HF_TOKEN")

def load_llm():
    llm = HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.5,
        max_new_tokens=512,
        task="conversational",
    )
    return llm

# Step 2: Define custom prompt template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Step 3: Load vectorstore database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 4: Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 5: Query loop (or single query)
user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})

print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])

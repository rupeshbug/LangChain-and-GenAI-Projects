from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.2, convert_system_message_to_human=True)

embeddings = HuggingFaceEmbeddings()

vectordb_file_path="faiss_index"

def create_vector_db():
    loader = CSVLoader('dermatology_q&a.csv', source_column='questions')
    data = loader.load() 
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings, allow_dangerous_deserialization=True)
    vectordb.save_local(vectordb_file_path)
    
def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)
    
    retriever = vectordb.as_retriever(score_threshold=0.7)
    
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""
    
    prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
    )
    
    chain = RetrievalQA.from_chain_type(
    chain_type="stuff",
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
    )
    
    return chain
    
    
    
if __name__ == "__main__":
    chain = get_qa_chain()
    
    print(chain.invoke("What causes melanoma?"))

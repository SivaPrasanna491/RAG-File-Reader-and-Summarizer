import os
import sys
import uvicorn 

from fastapi import FastAPI, File, UploadFile
from src.exception import CustomException
from src.logger import logging
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTraining
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader, UnstructuredExcelLoader, CSVLoader, DocxLoader
from langchain_community.vectorstores import Chroma, FAISS, LanceDB

app = FastAPI(
    title='RAG server',
    version='1.0',
    description='A simple RAG server'
)

document_loaders = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".csv": CSVLoader,
    ".docs": DocxLoader
}

vector_db = {
    ".txt": Chroma,
    ".pdf": FAISS,
    ".xlsx": LanceDB,
    ".csv": FAISS,
    ".docs": Chroma
}

vector_embeddings = {
    ".txt": "nomic-embed-text:v1.5",
    ".pdf": "snowflake-arctic-embed:335m",
    ".xlsx": "nomic-embed-text:v1.5",
    ".csv": "nomic-embed-text:v1.5",
    ".docs": "snowflake-arctic-embed:335m"
}

models = {
    ".txt": "llama2",
    ".pdf": "mistral",
    ".xlsx": "qwen3:0.6b",
    ".csv": "gemma3:270m",
    ".docs": "all-minilm:22m"
}

@app.post("/upload")
async def uploadFile(file: UploadFile):
    global db, temp_path
    os.makedirs("temp", exist_ok=True)
    
    temp_path = f"temp/{file.filename}"
    
    with open(temp_path, 'wb') as f:
        content = await file.read()
        f.write(content)
        
    ingestion_obj = DataIngestion(file_name=temp_path, loaders=document_loaders)
    documents = ingestion_obj.loadFile()
    
    transformation_obj = DataTransformation(documents=documents, file_name=temp_path, databases=vector_db, embeddings=vector_embeddings)
    db = transformation_obj.transformDocuments()
    
    os.remove(temp_path)
    return {"status": "uploaded"}

def query_rag(input_dict):
    query = input_dict['query']
    trainer_obj = ModelTraining(db=db, query=query, file_name=temp_path, models=models)
    return trainer_obj.getContext()
    
    
query_chain = RunnableLambda(query_rag)

add_routes(
    app,
    query_chain,
    path='/query'
)

if __name__=="__main__":
   uvicorn.run(app,host="localhost",port=8000)
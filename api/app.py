import os
import sys
import uvicorn
import asyncio
import uuid
from typing import Optional

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTraining
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from src.exception import CustomException
from src.logger import logging
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import (
    TextLoader, PyMuPDFLoader, UnstructuredExcelLoader, 
    CSVLoader, UnstructuredWordDocumentLoader
)
from langchain_community.vectorstores import Chroma, FAISS, LanceDB

# Set maximum file size to 1GB
MAX_FILE_SIZE = 1 * 1024 * 1024 * 1024  # 1GB in bytes

app = FastAPI(
    title='RAG server',
    version='1.0',
    description='A simple RAG server with 1GB file upload support'
)

document_loaders = {
    ".txt": TextLoader,
    ".pdf": PyMuPDFLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".csv": CSVLoader,
    ".docx": UnstructuredWordDocumentLoader
}

vector_db = {
    ".txt": Chroma,
    ".pdf": FAISS,
    ".xlsx": LanceDB,
    ".csv": FAISS,
    ".docx": Chroma
}

vector_embeddings = {
    ".txt": "nomic-embed-text:v1.5",
    ".pdf": "snowflake-arctic-embed:335m",
    ".xlsx": "nomic-embed-text:v1.5",
    ".csv": "nomic-embed-text:v1.5",
    ".docx": "snowflake-arctic-embed:335m"
}

models = {
    ".txt": "allam-2-7b",
    ".pdf": "llama-3.1-8b-instant",
    ".xlsx": "llama-3.1-8b-instant",
    ".csv": "groq/compound",
    ".docx": "allam-2-7b"
}

# Global variables
db = None
temp_path = None
processing_status = {}

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "running",
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
        "max_file_size_gb": MAX_FILE_SIZE / (1024 * 1024 * 1024)
    }

@app.post("/upload")
async def uploadFile(file: UploadFile = File(...)):
    """
    Upload large files (up to 1GB) with chunked reading
    """
    try:
        global db, temp_path
        
        os.makedirs("temp", exist_ok=True)
        temp_path = f"temp/{uuid.uuid4()}{os.path.splitext(file.filename)[1].lower()}"
        
        file_size = 0
        chunk_size = 10 * 1024 * 1024  # 10MB chunks for faster processing
        
        logging.info(f"Starting upload: {file.filename}")
        
        # Write file in chunks
        with open(temp_path, 'wb') as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                
                file_size += len(chunk)
                
                # Check size limit
                if file_size > MAX_FILE_SIZE:
                    f.close()
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024**3):.1f}GB. "
                               f"Your file is {file_size / (1024**3):.2f}GB"
                    )
                
                f.write(chunk)
                
                # Log progress for large files
                if file_size % (100 * 1024 * 1024) == 0:  # Log every 100MB
                    logging.info(f"Uploaded: {file_size / (1024**2):.1f}MB")
        
        logging.info(f"File saved successfully: {file_size / (1024**2):.2f}MB")
        
        # Data Ingestion
        logging.info("Starting document ingestion...")
        ingestion_obj = DataIngestion(file_name=temp_path, loaders=document_loaders)
        documents = ingestion_obj.loadFile()
        logging.info(f"Loaded {len(documents)} documents")
        
        # Data Transformation
        logging.info("Starting document transformation and embedding...")
        transformation_obj = DataTransformation(
            documents=documents, 
            file_name=temp_path, 
            databases=vector_db, 
            embeddings=vector_embeddings
        )
        db = transformation_obj.transformDocuments()
        logging.info("Vector DB created successfully")
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logging.info("Temp file cleaned up")
        
        return {
            "status": "success",
            "message": "File uploaded and processed successfully",
            "file_size_mb": round(file_size / (1024**2), 2),
            "documents_count": len(documents),
            "filename": file.filename
        }
    
    except HTTPException as he:
        raise he
    except Exception as e:
        # Clean up on error
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        logging.error(f"Upload error: {str(e)}")
        raise CustomException(e, sys)

@app.post("/upload-background")
async def uploadFileBackground(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload large files in background (for very large files > 100MB)
    Returns immediately and processes asynchronously
    """
    try:
        os.makedirs("temp", exist_ok=True)
        temp_path = f"temp/{file.filename}"
        job_id = f"{file.filename}_{os.getpid()}"
        
        # Save file first
        file_size = 0
        chunk_size = 10 * 1024 * 1024  # 10MB chunks
        
        with open(temp_path, 'wb') as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                file_size += len(chunk)
                
                if file_size > MAX_FILE_SIZE:
                    f.close()
                    os.remove(temp_path)
                    raise HTTPException(status_code=413, detail="File too large")
                
                f.write(chunk)
        
        # Add processing to background
        processing_status[job_id] = {"status": "processing", "progress": 0}
        background_tasks.add_task(
            process_file_background, 
            temp_path, 
            job_id,
            file_size
        )
        
        return {
            "status": "accepted",
            "job_id": job_id,
            "message": "File uploaded. Processing in background.",
            "check_status_url": f"/status/{job_id}"
        }
    
    except Exception as e:
        logging.error(f"Background upload error: {str(e)}")
        raise CustomException(e, sys)

async def process_file_background(file_path: str, job_id: str, file_size: int):
    """Background task to process large files"""
    try:
        global db, temp_path
        temp_path = file_path
        
        processing_status[job_id]["progress"] = 25
        
        # Ingestion
        ingestion_obj = DataIngestion(file_name=file_path, loaders=document_loaders)
        documents = ingestion_obj.loadFile()
        
        processing_status[job_id]["progress"] = 50
        
        # Transformation
        transformation_obj = DataTransformation(
            documents=documents,
            file_name=file_path,
            databases=vector_db,
            embeddings=vector_embeddings
        )
        db = transformation_obj.transformDocuments()
        
        processing_status[job_id]["progress"] = 90
        
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
        
        processing_status[job_id] = {
            "status": "completed",
            "progress": 100,
            "file_size_mb": round(file_size / (1024**2), 2),
            "documents": len(documents)
        }
        
    except Exception as e:
        processing_status[job_id] = {
            "status": "failed",
            "error": str(e)
        }
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Check processing status for background uploads"""
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job not found")
    return processing_status[job_id]

def query_rag(input_dict):
    try:
        if db is None:
            raise ValueError("No database loaded. Please upload a file first.")
        
        query = input_dict.get('query', '')
        if not query:
            raise ValueError("Query cannot be empty")
        
        trainer_obj = ModelTraining(
            db=db, 
            query=query, 
            file_name=temp_path, 
            models=models
        )
        return trainer_obj.getContext()
    
    except Exception as e:
        logging.error(f"Query error: {str(e)}")
        raise CustomException(e, sys)

query_chain = RunnableLambda(query_rag)

add_routes(app, query_chain, path='/query')

if __name__ == "__main__":
    # Configure uvicorn for large file uploads
    uvicorn.run(
        app, 
        host="0.0.0.0",  # Changed to 0.0.0.0 for external access
        port=8000,
        timeout_keep_alive=300,  # 5 minutes
        limit_concurrency=1000,
        limit_max_requests=10000
    )
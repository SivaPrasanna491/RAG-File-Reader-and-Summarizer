import os
import sys

from src.exception import CustomException
from src.logger import logging
from langchain_community.text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS, LanceDB
from langchain_ollama import OllamaEmbeddings

class DataTransformation():
    def __init__(self, documents, file_name):
        self.documents = documents
        self.file_name = file_name
    
    def transformDocuments(self):
        try:
            ext = os.path.splitext(self.file_name)
            if ext[1] == '.txt':
                splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 500)
                docs = splitter.split_documents(self.documents)
                logging.info("Dividing the documents into chunks is completed successfully")
                db = Chroma.from_documents(documents=docs, embedding=OllamaEmbeddings(model='nomic-embed-text:v1.5'))
                logging.info("Storing the chunked documents into vector store completed successfully")
                return db
            
            elif ext[1] == '.pdf':
                splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 500)
                docs = splitter.split_documents(self.documents)
                logging.info("Dividing the documents into chunks is completed successfully")
                db = FAISS.from_documents(documents=docs, embedding=OllamaEmbeddings(model='snowflake-arctic-embed:335m'))
                logging.info("Storing the chunked documents into vector store completed successfully")
                return db
            
            splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 500)
            docs = splitter.split_documents(self.documents)
            logging.info("Dividing the documents into chunks is completed successfully")
            db = LanceDB.from_documents(documents=docs, embedding=OllamaEmbeddings(model='nomic-embed-text:v1.5'))
            logging.info("Storing the chunked documents into vector store completed successfully")
            return db
        
        except Exception as e:
            raise CustomException(e, sys)
    
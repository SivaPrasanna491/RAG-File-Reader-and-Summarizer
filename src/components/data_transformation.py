import os
import sys

from src.exception import CustomException
from src.logger import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from src.utils import get_file_type

class DataTransformation():
    def __init__(self, documents, file_name, databases, embeddings):
        self.documents = documents
        self.file_name = file_name
        self.databases = databases
        self.embeddings = embeddings
    
    def transformDocuments(self):
        try:
            ext = os.path.splitext(self.file_name)
            extension = ext[1]
            
            if extension == '':
                extension = get_file_type(file_path=self.file_name)
                
            docs = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500).split_documents(self.documents)
            logging.info("Documents splitting done successfully")
            if(len(docs) > 0):
                try:
                    db = self.databases.get(extension).from_documents(documents=docs[:20], embedding=OllamaEmbeddings(model=self.embeddings.get(extension)))
                    logging.info("Documents splitting done successfully")
                    logging.info("Chunks stored in vector database successfully")
                    return db
                except Exception as e:
                    raise CustomException(e, sys)
        except Exception as e:
            raise CustomException(e, sys)
    
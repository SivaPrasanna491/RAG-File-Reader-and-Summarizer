import os
import sys

from src.exception import CustomException
from src.logger import logging
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredExcelLoader

class DataIngestion:
    def __init__(self, file_name):
        self.file_name = file_name
        
    def loadFile(self):
        try:    
            ext = os.path.splitext(self.file_name)
            logging.info("Extension of the file extracted successfully")
            
            if ext[1] == ".txt":
                loader = TextLoader(file_path = self.file_name)
                logging.info("File loaded successfully")
                return loader.load()
            
            elif ext[1] == ".pdf":
                loader = PyPDFLoader(file_path = self.file_name)
                logging.info("File loaded successfully")
                return loader.load()
            
            loader = UnstructuredExcelLoader(file_path = self.file_name)
            logging.info("File loaded successfully")
            return loader.load()
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    ingestion_obj = DataIngestion("Unit 5 pdf.pdf")
    print(ingestion_obj.loadFile())
    
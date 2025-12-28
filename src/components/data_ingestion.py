import os
import sys

from src.exception import CustomException
from src.logger import logging
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredExcelLoader
from src.utils import get_file_type

class DataIngestion:
    def __init__(self, file_name, loaders):
        self.file_name = file_name
        self.loaders = loaders
        
    def loadFile(self):
        try:    
            ext = os.path.splitext(self.file_name)
            extension = ext[1]
            
            if(extension == ''):
                extension = get_file_type(file_path=self.file_name)
            logging.info("Extension of the file extracted successfully")
            
            loader = self.loaders.get(extension)
            docs = loader(self.file_name).load()
            logging.info("Loading the data source done successfully")
            return docs
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    ingestion_obj = DataIngestion("Unit 5 pdf.pdf")
    print(ingestion_obj.loadFile())
    
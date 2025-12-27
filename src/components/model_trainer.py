import os
import sys

from src.exception import CustomException
from src.logger import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import Ollama
from langchain_classic.chains import create_retrieval_chain
from src.utils import get_file_type

class ModelTraining():
    def __init__(self, db, query, file_name):
        self.db = db
        self.query = query
        self.file_name = file_name
    
    def getContext(self):
        try:
            prompt = ChatPromptTemplate.from_template("""
            You are a helpful assistant. You help the users to get their queries clarified. Answer to the user in very friendly, professional way don't 
            answer the user in a very rude way. The user is gonna attach you a file it may be of any extension lile pdf/txt\excel etc. And they are gonna
            ask you the queries based on that file only sometimes they might ask you to summarize the file. I am gonna provide you some steps like how to give 
            answer the user in a very friendly and professional way. Follow the below steps as it is don't miss any step.

            Step-1: Read the file very carefully and read it twice to understand the context, and main keywords which are present in the file.
            Step-2: Answer the user query's by understanding the query twice and give the answer to the user's query based on context. Don't give random
            answers give them based on the context.

            Remember answer the user in a very friendly and professional way.

            <context>
            {context}
            </context>

            Question: {input}""")
            
            ext = os.path.splitext(self.file_name)
            extension = ext[1]
            
            if extension == '':
                extension = get_file_type(file_path=self.file_name)
                
            if extension == '.txt':
                llm = Ollama(model='llama2')
                document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
                logging.info("Chain created successfully")
                retriever = self.db.as_retriever()
                logging.info("Retriever created successfully")
                retrieverChain = create_retrieval_chain(retriever, document_chain)
                logging.info("Chain and Retriever combined successfully")
                reponse = retrieverChain.invoke(
                    {
                        "input": self.query
                    }
                )
                return reponse['answer']
            
            elif extension == '.pdf':
                llm = Ollama(model='mistral')
                document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
                logging.info("Chain created successfully")
                retriever = self.db.as_retriever()
                logging.info("Retriever created successfully")
                retrieverChain = create_retrieval_chain(retriever, document_chain)
                logging.info("Chain and Retriever combined successfully")
                reponse = retrieverChain.invoke(
                    {
                        "input": self.query
                    }
                )
                return reponse['answer']
            
            llm = Ollama(model='qwen3:0.6b')
            document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
            logging.info("Chain created successfully")
            retriever = self.db.as_retriever()
            logging.info("Retriever created successfully")
            retrieverChain = create_retrieval_chain(retriever, document_chain)
            logging.info("Chain and Retriever combined successfully")
            reponse = retrieverChain.invoke(
                {
                    "input": self.query
                }
            )
            return reponse['answer']
        
        except Exception as e:
            raise CustomException(e, sys)
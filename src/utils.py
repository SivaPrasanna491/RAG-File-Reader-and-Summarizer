import magic
import os
import sys

def get_file_type(file_path):
    mime = magic.from_file(file_path, mime=True)
    
    if mime == "application/pdf":
        return ".pdf"
    
    elif mime == "text/plain":
        return ".txt"
    
    elif mime in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 
                  'application/vnd.ms-excel']:
        return '.xlsx'
    
    else:
        return None
    
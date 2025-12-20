import os
import sys
import logging
from datetime import datetime
from pythonjsonlogger import jsonlogger
from app.config import ROOT_DIR

LOG_FILE_PATH = os.getenv("LOG_FILE_PATH")

def resolve_log_file_path() -> str:
    if LOG_FILE_PATH: return LOG_FILE_PATH
    
    # Define a pasta de logs em data/logs
    log_dir = os.path.join(ROOT_DIR, "data", "logs")
    
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except Exception as e:
            print(f"Erro ao criar pasta de logs: {e}")
            # Fallback
            return os.path.join(ROOT_DIR, "backend", "backend.out.log")

    return os.path.join(log_dir, "backend.out.log")

CURRENT_LOG_FILE = resolve_log_file_path()

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        if not log_record.get('timestamp'):
            now = datetime.fromtimestamp(record.created) if hasattr(record, 'created') else datetime.now()
            log_record['timestamp'] = now.strftime('%Y-%m-%dT%H:%M:%S')
        if not log_record.get('level'):
            log_record['level'] = record.levelname.upper() if record.levelname else 'INFO'

def setup_logging():
    log_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(CURRENT_LOG_FILE)
    
    formatter = CustomJsonFormatter(fmt='%(timestamp)s %(level)s %(name)s %(message)s %(module)s %(funcName)s %(lineno)d')
    
    log_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    app_logger = logging.getLogger("genexus_ai") # Nome fixo para facilitar filtro
    app_logger.setLevel(logging.INFO)
    app_logger.handlers = [file_handler]
    app_logger.propagate = False 
    
    # Intercepta bibliotecas
    for lib in ['uvicorn', 'uvicorn.access', 'uvicorn.error', 'fastapi']:
        l = logging.getLogger(lib)
        l.handlers = [file_handler]
        l.propagate = False
    
    # Root Logger
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [log_handler]
    
    return app_logger

logger = setup_logging()
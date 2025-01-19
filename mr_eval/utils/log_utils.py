import logging, os
from accelerate import Accelerator
from accelerate.logging import get_logger as get_accelerator_logger

current_file_path = os.path.dirname(os.path.abspath(__file__))
default_log_dir = f"{current_file_path}/../scripts/logs/generated"

def get_logger(name, level=logging.INFO, log_dir=default_log_dir):
    try:
        logger = get_accelerator_logger(name)
    except:
        print("Accelerator is not available, using the default logger.")
        logger = logging.getLogger(name)
    logger.setLevel(level)
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    # 创建文件处理器
    file_handler = logging.FileHandler(f"{log_dir}/{name}.log", mode='w')
    file_handler.setLevel(logging.DEBUG)

    
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    try:
        logger.logger.addHandler(console_handler)
        logger.logger.addHandler(file_handler)
    except:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger



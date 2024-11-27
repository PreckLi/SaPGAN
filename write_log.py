import logging
import os
import time

def write_logger(args, message):
    log_dir = '/home/likunhao/python_projects/spliteroberta_tpgan/exp_logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_path = os.path.join(log_dir, f'{args.llm}_{args.dataset}.log')
    handler = logging.FileHandler(log_file_path, encoding='utf-8')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logger.info(f'{formatted_time}\nArgs: {args}\nMessage: {message}')
    handler.close()
    logger.removeHandler(handler)

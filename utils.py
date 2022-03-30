import random
import jieba
import torch
import numpy as np
from pathlib import Path
from ckiptagger import data_utils
from termcolor import colored

DATA_PATH = Path(__file__).parent / "data"
EXPORT_PATH = (Path(__file__).parent / "export")

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def set_up():
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    EXPORT_PATH.mkdir(parents=True, exist_ok=True)
    ckip_check()
    set_seed(42)

def ckip_check():
    check_list = [
        'embedding_character', 
        'embedding_word',
        'model_ner',
        'model_pos',
        'model_ws'
    ]
    
    check = True
    
    for i in check_list:
        data_exists = (DATA_PATH / i).exists()
        print(
            colored(data_exists, 'blue') if data_exists else colored(data_exists, 'red'), 
            i
        )
        if not data_exists:
            check = False
            
        
    if not check:
        print("Lack of CKIP data, Start download...")
        data_utils.download_data_gdown("./")
        print("CKIP Data download complete.")    
        return
    
    print("CKIP Data validation complete.")
    
def clean_text(text):
    stoptext = open(DATA_PATH / 'stopword.txt', encoding='utf-8').read()
    stopwords = stoptext.split('\n')
    words = jieba.lcut(text)
    words = [w for w in words if w not in stopwords]
    return ' '.join(words)
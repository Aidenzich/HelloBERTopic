import os
from xmlrpc.client import Boolean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 關閉TF警告訊息
import pandas as pd
from bertopic import BERTopic
from ckiptagger import construct_dictionary, WS
from transformers import AutoModelForTokenClassification
from utils import EXPORT_PATH, set_up, DATA_PATH
from halo import Halo
from termcolor import colored
import argparse
import pickle

parser = argparse.ArgumentParser(description="Hello BERTopics")
parser.add_argument('--topic_num', type=int, default=10, help='設置要取出頻率排名前幾的topics')
parser.add_argument('--keyword_file', type=str, default="keys.txt", help='設置讀取keyword檔案名稱')
parser.add_argument('--model_name', type=str, default="ckiplab/bert-base-chinese-ws", help="設置HuggingFace的PretrainModel名稱")
parser.add_argument('--data_file', type=str, default="example_data.csv", help="設置資料讀取位置")
parser.add_argument('--word_sentence_cache', type=Boolean, default=False, help="是否讀取斷詞快取")


if __name__ == "__main__":
    args = parser.parse_args()
        
    set_up()
    
    # 設定 Huging Face Pretrained Model
    MODEL_NAME = args.model_name
    top_n_topics = args.topic_num
            
    # 讀取資料
    df = pd.read_csv(DATA_PATH / args.data_file)     
    sentence_list = df["description"].tolist()  # 我們取原始資料中的'description'欄位來當作訓練資料    
    timestamps = df.year.tolist()               # 讀取data.csv檔案中的 year 資料，作為我們的timestamp
    
    # 取出斷詞關鍵字
    keysfile = DATA_PATH / args.keyword_file
    with open(keysfile) as file:
        lines = file.read().splitlines() 

    ws_cache_path = EXPORT_PATH / 'word_sentence_cache.pkl'
    if args.word_sentence_cache and ws_cache_path.is_file():
        spinner = Halo(text='Load tokenized cache...', spinner='dots')
        spinner.start()
        print("Loading word sentence cache...")
        word_sentence_list = pickle.load(open(ws_cache_path, 'rb'))
        spinner.stop()
    
    else:
        # 讀取 CKIP 斷詞模型
        ws = WS(str(DATA_PATH))
        
        # 建立使用者字典 (幫助斷詞出關鍵字)
        keydict = { l: 1 for l in lines}
        dictionary = construct_dictionary(keydict)
                
        
        spinner = Halo(text='Tokenizing with CKIP-Tagger', spinner='dots')
        spinner.start()
        
        # 開始斷詞
        word_sentence_list = ws(
            sentence_list,
            sentence_segmentation = True,
            segment_delimiter_set = {",", "。", ":", "?", "!", ";"},
            recommend_dictionary = dictionary # 加入斷詞字典
        )
        
        spinner.stop()
            
        pickle.dump(word_sentence_list, open(ws_cache_path, 'wb'))
    
        
    
    # 轉換為BERTopic 可接受格式
    ws = [ " ".join(w) for w in word_sentence_list]
    print(colored(f"BERTopics Input Showcase: \n [ { ws[0] }]", 'blue'))
    
    spinner = Halo(text='Loading HagingFace Pretrained Model', spinner='dots')
    spinner.start()
    
    # 讀取 Hugingface Pretrained Model
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    spinner.stop()
    # 建立 BERTopic
    topic_model = BERTopic(
        language="chinese", 
        embedding_model=model,  
        verbose=True,
    )
    
    # 訓練並產生資料
    topics, probs = topic_model.fit_transform(ws)
    # 產生資料時間資料
    topics_over_time = topic_model.topics_over_time(ws, topics, timestamps, nr_bins=20)
    
    # 各 Topic TF-IDF 關鍵字直方圖
    bar_fig = topic_model.visualize_barchart(
        top_n_topics=top_n_topics,
        width=230,
    )
    
    # 各 Topic 向量分佈圖
    topic_fig = topic_model.visualize_topics(
        top_n_topics=top_n_topics,
        width=1000, 
        
    )
    
    # 各 Topic 向量分佈圖
    tot_fig = topic_model.visualize_topics_over_time(
        topics_over_time,
        top_n_topics=top_n_topics, 
        width=1000
    )
    
    # 儲存成 html 檔案，供前端展示使用
    bar_fig.write_html(EXPORT_PATH / "bar_fig.html")
    topic_fig.write_html(EXPORT_PATH / "topic_fig.html")
    tot_fig.write_html(EXPORT_PATH / "tot_fig.html")
    print('Done')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 關閉TF警告訊息
import pandas as pd
from bertopic import BERTopic
from ckiptagger import construct_dictionary, WS
from transformers import AutoModelForTokenClassification
from utils import EXPORT_PATH, set_up, DATA_PATH
from halo import Halo
from termcolor import colored

if __name__ == "__main__":
    
    
    set_up()
    
    # 設定 Huging Face Pretrained Model
    MODEL_NAME = "ckiplab/bert-base-chinese-ws"        
    top_n_topics = 20
    
    # 讀取 CKIP 斷詞模型
    ws = WS(str(DATA_PATH))
    
    # 讀取資料
    df = pd.read_csv(DATA_PATH / "data.csv")
    df = df[["year", "name", "label", "description"]]
    
    # 取出斷詞關鍵字
    keysfile = DATA_PATH / "keys.txt"
    with open(keysfile) as file:
        lines = file.read().splitlines() 

    # 建立使用者字典 (幫助斷詞出關鍵字)
    keydict = { l: 1 for l in lines}
    dictionary = construct_dictionary(keydict)
    
    # 我們取原始資料中的'description'欄位來當作訓練資料
    sentence_list = df["description"].tolist()
    # 讀取data.csv檔案中的 year 資料，作為我們的timestamp
    timestamps = df.year.tolist()
    
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
    
    # 轉換為BERTopic 可接受格式
    ws = [ " ".join(w) for w in word_sentence_list]
    print(colored(f"BERTopics Input Showcase: \n [ { ws[0] }]", 'blue'))
    
    
    # 讀取 Hugingface Pretrained Model
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    
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
        width=200,
    )
    
    # 各 Topic 向量分佈圖
    topic_fig = topic_model.visualize_topics(
        top_n_topics=top_n_topics,
        width=1000, 
        height=600
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
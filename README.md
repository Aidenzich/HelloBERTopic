# helloBERTopic
本專案用來對 [BERTopic](https://github.com/MaartenGr/BERTopic) 進行一些應用，日後會針對[作者文章](https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6)與[論文](https://arxiv.org/abs/2203.05794)做一些快速的摘要與實驗

## 執行結果
若成功執行完`main.py` 檔案，會在export資料夾中產生以下檔案：
```
bar_fig.html
topic_fig.html
tot_fig.html
```

## 專案結構
```
.
├── .gitignore
├── README.md
├── data
│   ├── data.csv
│   ├── keys
│   │   └── test.ipynb
│   ├── keys.txt
│   └── stopword.txt
├── exp.ipynb
├── main.py
├── requirements.txt
└── utils.py
```
## 安裝環境
```
pip install -r requirements.txt
```

## 執行程式
```
python main.py
```
### 參數說明
```
Hello BERTopics

optional arguments:
  -h, --help            show this help message and exit
  --topic_num TOPIC_NUM
                        設置要分成幾個topic
  --keyword_file KEYWORD_FILE
                        設置讀取keyword檔案名稱
  --model_name MODEL_NAME
                        設置HuggingFace的PretrainModel名稱
  --data_file DATA_FILE
                        設置資料讀取位置
  --word_sentence_cache WORD_SENTENCE_CACHE
                        是否讀取斷詞快取(如果沒有cache會走Default流程)
```

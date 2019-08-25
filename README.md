中文电商评论生成
========
由关键词生成一句用户评论，模型结构为seq2seq with attention。


### 环境
python 3.6
tensorflow >= 1.10

### 运行方法
准备好训练数据，样例已经放在"works/word2sen/data"中。\
1.train:\
```python train.py```
2.infer:\
```python infer.py```

# DecoderLayerLM

这是自己写的一个语言模型

### 1. 程序文件

`config.py`: 参数选择

`dataset.py`: 数据集， 使用的是莎士比亚的作品

`model.py`: 模型主体

`model_chat.py`: 加载模型完成对话的文件

`train,py`: 训练文件， 必须先训练之后才可以加载文件

`./dat` : 整理的数据和缓存目录，这里以莎士比亚的诗歌为数据集

`./model`: 模型目录， 模型大小只有28mb

### 2. 使用方法

训练直接运行 `train,py`

对话请运行 `model_chat.py`

例子:
```
question = "  You were pretty lordings then? "
gen_chat(model, vocab, question, 1)
```
回答:

```
we were , fair queen , two lads that thought there was no more behind but such a day to-morrow as to-day , and to be boy eternal .
[41, 85, 4, 156, 172, 4, 208, 3218, 15, 284, 82, 68, 43, 62, 855, 30, 112, 14, 149, 373, 33, 644, 4, 8, 10, 27, 287, 2791, 5] len: 29
```

可以到 `./dat/Shakespeare_preprocessed.txt` 中查找上下文进行对比


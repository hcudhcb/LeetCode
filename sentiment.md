基于双向GRU的文本情感分类

代码是基于技术报告里的单向LSTM的情感分类的改进。

数据集来源：
Stanford Linguistic Data Consortium，http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip

改进点：
```
1.词向量维度从300改为50，word2vec的epoch次数从1000改为100
2.网络结构由单向LSTM改为双向GRU
3.epoch次数从1000改为100
```
效果：
```
整体来看：准确率，精确率，召回率，F1值都有一定的下降，
- loss: 0.3752 - acc: 0.8310 - recall: 0.8155 - precision: 0.8415 - f1: 0.8276
- val_loss: 0.3973 - val_acc: 0.8184 - val_recall: 0.8028 - val_precision: 0.8296 - val_f1: 0.8154
初步分析主因是可能由于词向量表达上的维度降低及训练次数减少，以及随着训练次数增加学习率降低太多导致的可能陷入了局部最优。
优势：改为双向模型之后收敛速度加快，大约epoch到50次即可达到目前参数上的最优值，说明双向模型更容易catch上下文语境，单向的LSTM一般要到200次左右才能将准确率达到80%。
```
后续：
```
1.词向量训练和base保持一致，改为300维和1000次。
2.epoch到40次之后，衰减一次学习率，衰减系数为0.2，目前是0.1。
```
代码结构：
```
|----data_process(数据预处理包)
|----|----data_process.py(数据预处理)
|----|----extract_sentence.py(原始数据文件的读取及处理后写入)
|
|----word2vec(词向量模型训练)
|----|----word2vector_build.py(训练模型及模型保存)
|
|----GRU
|----|----BiGRU.py(双向GRU模型训练)
|
|----__main__.py(主函数，控制代码执行顺序)
```
环境依赖：
```
python3.7
gensim
nltk
tensorflow
keras
numpy
```
```
cuda10.1(在cuda9.0的机器上无法使用gensim不知道为何，换到cuda10就好了)
TITAN RTX
```
运行步骤：
```
export dataset="path where the dataset"
export processes="path where the processed sentences you want to save"
export word2vec="path where you want to save the word2vec model"
export label="path where the label of npy file"
python __main__.py --dataset=$dataset --processed=$processed --word2vec=$word2vec --label=$label
```

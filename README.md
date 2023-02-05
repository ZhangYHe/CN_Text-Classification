# CN_text-classification

基于pytorch的中文文本分类   

BIT《自然语言理解初步》课程大作业2   

使用python进行代码编写，使用Google Colab与PyCharm进行调试与训练。采用Pytorch搭建Bert模型，使用[中文预训练模型权重BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)，预训练数据包括中文维基百科，其他百科、新闻、问答等数据，总词数达5.4B。训练数据使用[THUCNews新闻文本分类数据集](http://thuctc.thunlp.org/)的子集。

- chinese_roberta_wwm_ext_pytorch为预训练的Bert模型
- data文件夹中的ckpt存储训练好的模型
- cnews文件夹存储经过处理的THUCNews新闻文本分类数据集
- BIT2022_NLU_h2.ipynb文件用于功能演示。

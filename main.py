#!pip install pytorch_pretrained_bert
# coding: UTF-8
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig, BertAdam
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, TensorDataset, RandomSampler, DataLoader, SequentialSampler


# 定义模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 加载预训练的bert模型权重
        self.bert = BertModel.from_pretrained(bert_path)
        # 对于每个参数均求梯度
        for param in self.bert.parameters():
            param.requires_grad = True
            # 线性层求10个类别的概率
        self.fc = nn.Linear(768, 10)

    def forward(self, x):
        sentence = x[0]
        types = x[1]
        mask = x[2]
        _, pooled = self.bert(sentence, token_type_ids=types,
                              attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out


# 预测句子类别
def predict_type(sentence, path):
    # 加载最优的模型
    model.load_state_dict(torch.load(path))

    sentence = tokenizer.tokenize(sentence)
    tokens = ["[CLS]"] + sentence + ["[SEP]"]
    ids = tokenizer.convert_tokens_to_ids(tokens)
    types = [0] * (len(ids))
    masks = [1] * len(ids)
    # 与pad_size比较，进行切断或填补
    if len(ids) < pad_size:
        types = types + [1] * (pad_size - len(ids))
        masks = masks + [0] * (pad_size - len(ids))
        ids = ids + [0] * (pad_size - len(ids))
    else:
        types = types[:pad_size]
        masks = masks[:pad_size]
        ids = ids[:pad_size]

    ids, types, masks = torch.LongTensor(np.array(ids)), torch.LongTensor(np.array(types)), torch.LongTensor(
        np.array(masks))

    y_pred = model([ids.reshape(1, -1), types.reshape(1, -1), masks.reshape(1, -1)])

    return sentence_types[torch.argmax(y_pred)]


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':

    # 数据处理
    # word_ids存放词语的id
    # word_types中0、1区分不同句子
    # word_masks为attention中的掩码，0表示padding
    word_ids = []
    word_types = []
    word_masks = []
    labels = []
    pad_size = 50
    # 句子类别共10种
    sentence_types = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    len_dict = dict()

    # data_path为数据集路径，bert_path为预训练模型权重路径,ckpt_path为模型路径
    data_path = "./data/cnews/"
    bert_path = "./chinese_roberta_wwm_ext_pytorch/"
    ckpt_path = "./data/ckpt/"

    # 初始化分词器，使用预训练的vocab
    tokenizer = BertTokenizer(vocab_file=bert_path + "vocab.txt")

    # 若有可用设备则使用cuda进行计算，否则cpu计算
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    model = Model().to(DEVICE)

    printf("-> Please enter the sentence.")
    sentence = input()

    types = predict_type(sentence, ckpt_path + "roberta_model.pth")
    print(types)

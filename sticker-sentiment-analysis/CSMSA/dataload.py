import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random
import argparse
from transformers import BertTokenizer, BertModel, BertConfig, AutoTokenizer
import glob
#from paddleocr import PaddleOCR

dess = {}

with open('/home/ycshi/sticker-sentiment/shared_dataset/dess-yi/odocrti.txt', 'r', encoding='utf-8') as f:
    datas = f.readlines()
for data in datas:
    dess[data.split('&&&')[0]] = data.split('&&&')[1].replace('\n', '')

CFG = {  # 训练的参数配置
    'seed': 2023,
    'context_model': r'/home/ycshi/huggingface/hflchinese-roberta-wwm-ext',  # 预训练模型 'hfl/chinese-roberta-wwm-ext'
    'text_model': r'/home/ycshi/huggingface/hflchinese-roberta-wwm-ext',  # 预训练模型
    'epochs': 200,
    'max_len': 250, # 比较重要，显存不够改这个！！！
    'train_bs': 32,  # batch_size，可根据自己的显存调整
    'valid_bs': 4,
    'lr': 1e-5,  # 学习率
    'num_workers': 0,
    'accum_iter': 1,  # 梯度累积，相当于将batch_size*2
    'weight_decay': 1e-5,  # 权重衰减，防止过拟合
    'device': 7,
}

tokenizer = AutoTokenizer.from_pretrained(CFG['context_model'])

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG['seed'])  # 固定随机种子

torch.cuda.set_device(CFG['device'])
device = torch.device(CFG['device'])

def get_all_file_paths(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths
'''
def load_sticker_features(pth):
    sticker_features = dict()
    filenames = glob.glob(os.path.join(pth, '*.pt'))
    for filename in filenames:
        sticker_id = filename.split('/')[-1].split('.')[0]
        sticker_features[sticker_id] = torch.load(filename)
    return sticker_features

all_sticker_features = load_sticker_features('/home/ycshi/sticker-intent1/data/sticker-features-Clip')

def get_data(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    sticker_id = list(set([i['sticker'] for i in data]))
    random.shuffle(sticker_id) # 随机打乱
    train,valid = [],[]
    rewrite = []

    ocr = PaddleOCR(use_angle_cls=True, lang="ch") # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
    for item in data:
        if item['sticker_class'] in [3,4,5,6]:
            img_path = '/home/ycshi/sticker-intent1/data/all_sticker/' + item['sticker'] 
            try:
                result = ocr.ocr(img_path+ '.png', cls=True)
            except:
                result = ocr.ocr(img_path+ '.webp', cls=True)
            result = result[0]
            txts = [line[1][0] for line in result]
            text = ''
            for i in range(len(txts)):
                text += txts[i]
            item['sticker_text'] = text
        else:
            item['sticker_text'] = ''
        if item['sticker'] in sticker_id[:int(len(sticker_id)*0.9)]:
            train.append(item)
        else:
            valid.append(item)
        rewrite.append(item)
        # 将rewrite的数据存储为方便查看的json文件，利用indent参数使json文件格式化
        with open('/home/bma/sticker-intent/data/rewrite.json', 'w', encoding='utf-8') as f:
            json.dump(rewrite, f, ensure_ascii=False, indent=4)
    return train,valid

# 判读是否存在train.json和valid.json，如果不存在则生成
if not os.path.exists('/home/bma/sticker-intent/data/train.json') or not os.path.exists('/home/bma/sticker-intent/data/valid.json'):
    train,valid = get_data('/home/bma/sticker-intent/data/all_data.json')

    # 将train和valid的数据存储为json文件
    with open('/home/bma/sticker-intent/data/train.json', 'w', encoding='utf-8') as f:
        json.dump(train, f, ensure_ascii=False)
    with open('/home/bma/sticker-intent/data/valid.json', 'w', encoding='utf-8') as f:
        json.dump(valid, f, ensure_ascii=False)

else:
    # 如果已经存在了，直接从json文件中读取
    with open('/home/bma/sticker-intent/data/train.json', encoding='utf-8') as f:
        train = json.load(f)
    with open('/home/bma/sticker-intent/data/valid.json', encoding='utf-8') as f:
        valid = json.load(f)
'''
# multimodal_intent_str2num = {'Complain':0}
# for i in range(len(train)):
    
#     intent = train[i]['multimodal_intent_label']
#     values = list(multimodal_intent_str2num.values())
#     keys = list(multimodal_intent_str2num.keys())
#     if intent not in list(keys):
#         # 字典赋值
#         multimodal_intent_str2num[intent] = max(values) + 1
# print(multimodal_intent_str2num)



class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        data = self.df[idx]
        return data
    
def collate_fn(batch):
    context_ids, context_masks, token_type_ids, emo_lable =[],[],[], []
    #multimodal_intent_str2num = {'Complain': 0, 'Praise': 1, 'Agree': 2, 'Compromise': 3, 'Query': 4, 'Joke': 5, 'Oppose': 6, 'Inform': 7, 'Ask for help': 8, 'Greet': 9, 'Taunt': 10, 'Introduce': 11, 'Guess': 12, 'Leave': 13, 'Advise': 14, 'Flaunt': 15, 'Criticize': 16, 'Thank': 17, 'Comfort': 18, 'Apologize': 19}
    for x in batch:
        #sf = all_sticker_features[x['sticker']].squeeze(0)
            
        # except:
        #     sf = np.zeros(2048, dtype=np.float32)
        
        #sticker_features.append(sf)
        text = dess[x['image']]
        
        # 将文本转化为token
        token = tokenizer(text,padding='max_length', truncation=True, max_length=CFG['max_len'],return_tensors='pt')
        context_ids.append(token['input_ids'][0])
        context_masks.append(token['attention_mask'][0])
        token_type_ids.append(token['token_type_ids'][0])

        emo_lable.append(x['image_label'])

        #intent_lable.append(multimodal_intent_str2num[x['multimodal_intent_label']])
    context_ids = torch.stack(context_ids)
    context_masks = torch.stack(context_masks)
    token_type_ids = torch.stack(token_type_ids)

    #sticker_features = torch.stack(sticker_features)
    emo_lables = torch.tensor(emo_lable)

    #intent_lables = torch.tensor(intent_lable)

    return context_ids, context_masks, token_type_ids, emo_lables 


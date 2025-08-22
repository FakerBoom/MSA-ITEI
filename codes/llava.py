from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import json
import os
processor = LlavaNextProcessor.from_pretrained("/home/ycshi/huggingface/llava-v1.6-mistral-7b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained("/home/ycshi/huggingface/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
model.to("cuda:1")

ocrs = {}

with open('/home/ycshi/sticker-sentiment/shared_dataset/ocr.txt','r',encoding='utf-8') as f1:
    datas = f1.readlines()
    
for data in datas:
    image_id = data.split('&&&')[0]
    text = data.split('&&&')[1].replace('\n','')
    ocrs[image_id] = text

# prepare image and text prompt, using the appropriate prompt template



image_s = os.listdir('/home/ycshi/sticker-sentiment/shared_dataset/sticker')
for imageid in image_s:
    image_file = '/home/ycshi/sticker-sentiment/shared_dataset/sticker/' + imageid
    image = Image.open(image_file)
    if imageid in ocrs:
        ocr = ocrs[imageid]
    else:
        ocr = ''
    if ocr == '':
        alls = '用一句话描述这张表情包的整体形象,图中没有文字&&用一句话描述这张表情包整体形象&&用一句话描述这张表情包,图中没有文字'
    else:
        alls = '用一句话描述这张表情包的整体形象,图中文字为‘'+ocr+'’&&用一句话描述这张表情包整体形象&&用一句话描述这张表情包,图中文字为‘'+ocr+'’'    
    qss = alls.split("&&")
    paths = {1: '/home/ycshi/sticker-sentiment/shared_dataset/dess-llava/odocr.txt',
             2:'/home/ycshi/sticker-sentiment/shared_dataset/dess-llava/od.txt',
             3:'/home/ycshi/sticker-sentiment/shared_dataset/dess-llava/ocr.txt'}
    num = 1
    print(qss)
    for qs in qss:
        path = paths[num]
        num += 1
        prompt = "[INST] <image>\n" + qs + " [/INST]"
        inputs = processor(prompt, image, return_tensors="pt").to("cuda:1")

    # autoregressively complete prompt
        output = model.generate(**inputs, max_new_tokens=100)
        out = processor.decode(output[0], skip_special_tokens=True).replace('\n','').replace(' ','').split('[/INST]')[1]
        print(out+ '\n')
        f = open(path,'a',encoding='utf-8')
        f.write(imageid + '&&&' + out + '\n')


'''
ocrs = {}

with open('/home/ycshi/sticker-sentiment/ourdata/rewrite.json','r',encoding='utf-8') as f1:
    datas = json.load(f1)
for data in datas:
    image_id = data['sticker']+'.png'
    text = data['sticker_text']
    text = text.replace('\n','')
    ocrs[image_id] = text

# prepare image and text prompt, using the appropriate prompt template



image_s = os.listdir('/home/ycshi/sticker-sentiment/shared_dataset/sticker')
for imageid in image_s:
    image_file = '/home/ycshi/sticker-sentiment/shared_dataset/sticker/' + imageid
    image = Image.open(image_file)
    if imageid in ocrs:
        ocr = ocrs[imageid]
    else:
        ocr = ''
    if ocr == '':
        alls = '用一句话描述这张表情包的整体形象和可能的情绪意图&&用一句话描述这张表情包表达的情绪和意图&&用一句话描述这张表情包可能的情绪意图,图中没有文字&&用一句话描述这张表情包的整体形象和可能表达的情绪意图,图中没有文字'
    else:
        alls = '用一句话描述这张表情包的整体形象和可能的情绪意图&&用一句话描述这张表情包表达的情绪和意图&&用一句话描述这张表情包可能的情绪意图,图中文字为‘'+ocr+'’&&用一句话描述这张表情包的整体形象和可能表达的情绪意图,图中文字为‘'+ocr+'’'    
    qss = alls.split("&&")
    paths = {1: '/home/ycshi/sticker-sentiment/shared_dataset/dess-llava/odti.txt',
             2:'/home/ycshi/sticker-sentiment/shared_dataset/dess-llava/ti.txt',
             3:'/home/ycshi/sticker-sentiment/shared_dataset/dess-llava/ocrti.txt',
             4: '/home/ycshi/sticker-sentiment/shared_dataset/dess-llava/odocrti.txt'}
    num = 1
    print(qss)
    for qs in qss:
        path = paths[num]
        num += 1
        prompt = "[INST] <image>\n" + qs + " [/INST]"
        inputs = processor(prompt, image, return_tensors="pt").to("cuda:1")

    # autoregressively complete prompt
        output = model.generate(**inputs, max_new_tokens=100)
        out = processor.decode(output[0], skip_special_tokens=True).replace('\n','').replace(' ','').split('[/INST]')[1]
        print(out+ '\n')
        f = open(path,'a',encoding='utf-8')
        f.write(imageid + '&&&' + out + '\n')
    
'''
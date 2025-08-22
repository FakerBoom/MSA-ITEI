from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import json
import os
processor = LlavaNextProcessor.from_pretrained("/home/ycshi/huggingface/llava-v1.6-mistral-7b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained("/home/ycshi/huggingface/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
model.to("cuda:7")


ocrs = []

with open('/home/ycshi/sticker-sentiment/ourdata/valid.json','r',encoding='utf-8') as f1:
    datas = json.load(f1)
for data in datas:
    image_id = data['sticker']+'.png'
    text = data['context']
    label = data['multimodal_intent_label']
    #text = text.replace('\n','')
    ocrs.append((image_id,text,label))

# prepare image and text prompt, using the appropriate prompt template



for (imageid,ocr,label) in ocrs:
    image_file = '/home/ycshi/sticker-sentiment/ourdata/all_sticker/' + imageid
    image = Image.open(image_file)
    qs = '分析表情包搭配上下文“'+ocr+'”表达了什么意图,答案从以下20个单词中选最恰当的一个，直接输出表达意图的单词：Complain, Praise, Agree, Compromise, Query, Joke, Oppose, Inform, Ask for help, Greet, Taunt, Introduce, Guess, Leave, Advise, Flaunt, Criticize, Thank, Comfort, Apologize'
    prompt = "[INST] <image>\n" + qs + " [/INST]"
    inputs = processor(prompt, image, return_tensors="pt").to("cuda:7")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=100)
    out = processor.decode(output[0], skip_special_tokens=True).replace('\n','').replace(' ','').split('[/INST]')[1]
    print(out+ '\n')
    f = open('/home/ycshi/sticker-sentiment/ourdata/results/intent/llavaIR.txt','a',encoding='utf-8')
    f.write(imageid + '的情感是&&&' + out + '***正确的答案应该是'+label+'\n')   
    

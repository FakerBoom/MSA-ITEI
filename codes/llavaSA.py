from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import json

processor = LlavaNextProcessor.from_pretrained("/home/ycshi/huggingface/llava-v1.6-mistral-7b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained("/home/ycshi/huggingface/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
model.to("cuda:1")

with open('/home/ycshi/sticker-sentiment/ourdata/valid.json', 'r', encoding='utf-8') as f:
    datas = json.load(f)

#f1 = open('/home/ycshi/sticker-sentiment/ourdata/emnlp2024-rebuttal/Qwen.txt', 'w', encoding='utf-8')

right1 = 0
wrong1 = 0
right2 = 0
wrong2 = 0

for data in datas:
    context = data['context']
    sticker = data['sticker']
    image = '/home/ycshi/sticker-sentiment/ourdata/all_sticker/'+sticker+'.png'
    image = Image.open(image)
    
    text = "[INST]What sentiment (0,1,2 stands for neutral, positive, negative, respectively) and intent (Use the following numbers to replace the corresponding intent labels: 'Complain' = 0, 'Praise' = 1, 'Agree' = 2, 'Compromise' = 3, 'Query' = 4, 'Joke' = 5, 'Oppose' = 6, 'Inform' = 7, 'Ask for help' = 8, 'Greet' = 9, 'Taunt' = 10, 'Introduce' = 11, 'Guess' = 12, 'Leave' = 13, 'Advise' = 14, 'Flaunt' = 15, 'Criticize' = 16, 'Thank' = 17, 'Comfort' = 18, 'Apologize' = 19) does context:"+context+\
    ",along with the sticker: <image> show? Output the numbers directly and separate with commas.For example, if the context and sticker together reflect the positive sentiment and Greet intent, just output '1,9'[/INST]"


    inputs = processor(text, image, return_tensors="pt").to("cuda:0")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=100)
    out = processor.decode(output[0], skip_special_tokens=True).replace('\n','').replace(' ','').replace('，',',').split('[/INST]')[1]
    print(out)
    predict_sentiment = int(out.split(',')[0])
    predict_intent = int(out.split(',')[1])
    sentiment = data['multimodal_sentiment_label']
    intent = data['multimodal_intent_label']
    if predict_sentiment == sentiment:
        right1 += 1
    else:
        wrong1 += 1
    if predict_intent == intent:
        right2 += 1
    else:
        wrong2 += 1
    acc1 = right1/(right1+wrong1)
    f11 = 2*right1/(2*right1+wrong1+wrong2)
    acc2 = right2/(right2+wrong2)
    f12 = 2*right2/(2*right2+wrong2+wrong1)
    print('Sentiment:',acc1,f11)
    print('Intent:',acc2,f12)
    '''
    text = "[INST]What sentiment (0,1,2 stands for neutral, positive, negative, respectively) does context:"+context+\
    ",along with the sticker: <image> show? Output the number directly. For example, if the context and sticker together reflect the positive sentiment, just output '1'。[/INST]"
    text1 = "[INST]What intent (Use the following numbers to replace the corresponding intent labels: 'Complain' = 0, 'Praise' = 1, 'Agree' = 2, 'Compromise' = 3, 'Query' = 4, 'Joke' = 5, 'Oppose' = 6, 'Inform' = 7, 'Ask for help' = 8, 'Greet' = 9, 'Taunt' = 10, 'Introduce' = 11, 'Guess' = 12, 'Leave' = 13, 'Advise' = 14, 'Flaunt' = 15, 'Criticize' = 16, 'Thank' = 17, 'Comfort' = 18, 'Apologize' = 19) does context:"+context+\
    ",along with the sticker: <image> show? Output the number directly. For example, if the context and sticker together reflect the Greet intent, just output '9'。[/INST]"
    inputs = processor(text, image, return_tensors="pt").to("cuda:1")
    output = model.generate(**inputs, max_new_tokens=100)
    out = processor.decode(output[0], skip_special_tokens=True).replace('\n','').replace(' ','').replace('，',',').split('[/INST]')[1]
    predict_sentiment = int(out)
    sentiment = data['multimodal_sentiment_label']
    if predict_sentiment == sentiment:
        right1 += 1
    else:
        wrong1 += 1
    inputs = processor(text1, image, return_tensors="pt").to("cuda:1")
    output = model.generate(**inputs, max_new_tokens=100)
    out = processor.decode(output[0], skip_special_tokens=True).replace('\n','').replace(' ','').replace('，',',').split('[/INST]')[1]
    predict_intent = int(out)
    intent = data['multimodal_intent_label']
    if predict_intent == intent:
        right2 += 1
    else:
        wrong2 += 1
    acc1 = right1/(right1+wrong1)
    f11 = 2*right1/(2*right1+wrong1+wrong2)
    acc2 = right2/(right2+wrong2)
    f12 = 2*right2/(2*right2+wrong2+wrong1)
    print('Sentiment:',acc1,f11)
    print('Intent:',acc2,f12)
    '''
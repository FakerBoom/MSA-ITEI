import os
import json
import shutil

dessi = {}
desst = {}
dessei = {}

with open('/home/ycshi/sticker-sentiment/ourdata/dess-llava/od.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split('&&&')
        dessi[line[0].split('.')[0]] = line[1].replace('\n', '')

with open('/home/ycshi/sticker-sentiment/ourdata/dess-llava/ocr.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split('&&&')
        desst[line[0].split('.')[0]] = line[1].replace('\n', '')

with open('/home/ycshi/sticker-sentiment/ourdata/dess-llava/ti.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split('&&&')
        dessei[line[0].split('.')[0]] = line[1].replace('\n', '')

dess = {}
for key in dessi:
    dess[key] = dessi[key] +  desst[key] +  dessei[key]

with open('/home/ycshi/sticker-sentiment/ourdata/valid.json', 'r', encoding='utf-8') as f:
    datas = json.load(f)
    captions = {}
    
    for data in datas:
        image = data['sticker']
        captions[image] = dess[image]

    with open('/home/ycshi/sticker-sentiment/ourdata/clipscore-data/jsons/i&t&ei.json', 'w', encoding='utf-8') as f1:
        json.dump(captions, f1, ensure_ascii=False)

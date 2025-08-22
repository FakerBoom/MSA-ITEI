import json
import os



with open('/home/ycshi/sticker-sentiment/ourdata/results/yi/odocr.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()
    
accs = []
n = 1
for i in data:
    if 'valid_loss:' in i and 'emo_acc:' in i:
        accs.append([n, i.split('emo_acc:')[1]])
    n += 1

#print top-3 accs
accs.sort(key=lambda x: x[1], reverse=True)
for i in range(10):
    print(accs[i])
import json
with open('/home/ycshi/sticker-sentiment/ourdata/train.json', 'r', encoding='utf-8') as f:
    datas = json.load(f)
    

with open('/home/ycshi/sticker-sentiment/ourdata/dess-llava/od.txt', 'r', encoding='utf-8') as f2:
    ods = f2.readlines()

od1 = {}
for i in ods:
    k, v = i.strip().replace("\n",'').split('&&&')
    k = k.split('.')[0]
    od1[k] = v

with open('/home/ycshi/sticker-sentiment/ourdata/dess-llava/ocrti.txt', 'r', encoding='utf-8') as f2:
    ods = f2.readlines()

od2 = {}
for i in ods:
    k, v = i.strip().replace("\n",'').split('&&&')
    k = k.split('.')[0]
    od2[k] = v


for data in datas:
    '''
    prompt = '原对话：“A：' + data["context"].replace("\t",'') + '\nB：[表情包]”包含了一张用“[表情包]”表示的表情包图片。'
    if data["sticker_text"] != "":
        prompt += '表情包中的文字是“' + data["sticker_text"] + '”。'
    prompt += '这张表情包里的图像和可能表达的情绪意图是：“' + od[data["sticker"]] + '”'
    prompt += '请基于上下文和给你的表情包视觉、情感意图信息，用一句合适的文本代替“[表情包]”，来重构原对话。直接输出重构后的对话，不要输出其他内容。'
    data["prompt"] = prompt
    '''
    new_dialogue = "A: " + data["context"].replace("\t",'') + "\nB: " + od1[data["sticker"]]+ od2[data["sticker"]]
    data["new_dialogue"] = new_dialogue

with open('/home/ycshi/sticker-sentiment/ourdata/dess-llava/train_wodr.json', 'w', encoding='utf-8') as f3:
    json.dump(datas, f3, ensure_ascii=False, indent=4)
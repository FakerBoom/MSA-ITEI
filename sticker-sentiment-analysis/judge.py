with open('/home/ycshi/sticker-sentiment/stickerSA/CSMSA/yi/easy.txt', 'r', encoding='utf-8') as f:
    datas = f.readlines()
right = 0
wrong = 0
for data in datas:
    x= data.split('的情感是&&&')[1].replace('\n','')
    predict = x.split('***正确的答案应该是')[0]
    answer = x.split('***正确的答案应该是')[1]
    if predict == answer:
        right += 1
    else:
        wrong += 1

print('accuracy:',right/(right+wrong))
print('f1:',right/(right+0.5*(right+wrong)))
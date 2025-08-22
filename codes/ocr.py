import shutil
import os
from paddleocr import PaddleOCR

images = os.listdir('/home/ycshi/sticker-sentiment/ourdata/all_sticker')

f = open('/home/ycshi/sticker-sentiment/ourdata/ocr.txt','w',encoding='utf-8')
ocr = PaddleOCR(use_angle_cls=True, lang="ch") # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`

for image in images:
    img_path = '/home/ycshi/sticker-sentiment/ourdata/all_sticker/' + image
    result = ocr.ocr(img_path, cls=True)
    result = result[0]
    if result == None:
        f.write(image + '&&&'  + '\n')
        continue
    txts = [line[1][0] for line in result]
    text = ''
    for i in range(len(txts)):
        text += txts[i]
    f.write(image + '&&&' + text + '\n')



import os
import json

def merge_json_files(input_folders, output_file):
    merged_data = []
    for folder in input_folders:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        merged_data.extend(data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

# 要合并的文件夹列表
input_folders = ['/home/ycshi/sticker-sentiment/shared_dataset/easy_task', '/home/ycshi/sticker-sentiment/shared_dataset/hard_task1',
                 '/home/ycshi/sticker-sentiment/shared_dataset/hard_task2','/home/ycshi/sticker-sentiment/shared_dataset/hard_task3']

# 合并后的输出文件
output_file = '/home/ycshi/sticker-sentiment/shared_dataset/merged.json'

# 执行合并操作
merge_json_files(input_folders, output_file)

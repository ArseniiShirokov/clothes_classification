import os
import json
from tqdm import tqdm
import pandas as pd
from random import shuffle
import sys
import shutil
from pathlib import Path

dataset_paths = [
        "/home/local/remote_files/Exchange/2022_01_28_reid_data/DataMiningVideosReIDBalanced/crops"
#         "/home/local/Attributes-pytorch/data/Datasets/Metro/MetroTest_part1"

]

for dataset_path in dataset_paths:
    if not os.path.isdir(dataset_path):
        print(f"Dataset {dataset_path} not found")
        
output_name = 'reid_data'

# Generate general markup fiel
json_markup = dict()

for i, dataset_path in enumerate(dataset_paths):
    images = list(Path(dataset_path).rglob("*.jpg"))
    for image in tqdm(images):
        json_markup[os.path.join(dataset_path, image)] = [-1]
    
# Convert json to .lst, format indx\t label_1\t .. label_n\t img_path
out_paths = []
imgs_paths = list(json_markup.keys())
print(len(imgs_paths))
lst_path = f'generated/{output_name}.lst'
out_paths.append(lst_path)
with open(lst_path, 'w+') as f:
    for indx, img_path in enumerate(tqdm(imgs_paths)):
        labels = map(str, json_markup[img_path])
        out = f'{indx}\t' + '\t'.join(labels) + '\t' + img_path + '\n'
        f.write(out)
        
json.dump(json_markup, open(f'generated/{output_name}.json', 'w'))
json.dump([len(imgs_paths)], open(f'generated/{output_name}.cnt', 'w'))

for lst_path in out_paths:
    os.system(f"python3 im2rec.py {lst_path} ./ --pack-label")
    


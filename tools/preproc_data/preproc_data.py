import os
import json
import shutil
from typing import List
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import pandas as pd
import sys


def filter_markup(markup: dict, images_dir: str) -> dict:
    """Filter markup items with unexisting images"""
    filter_part = os.listdir(images_dir)
    return {key: value for (key, value) in markup.items() if key in filter_part}


def union_markups(images_dir: List[str], markups: List[str]) -> dict:
    union_markup = dict()
    for i, image_dir in enumerate(images_dir):
        # Open json markup file, format {img_name: [label_1, .. label_n]}
        markup_path = os.path.join(image_dir, markups[i])
        # Markup file can be in image_dir or in image_dir/..
        if not os.path.exists(markup_path):
            markup_path = os.path.join(image_dir, "..", markups[i])
            if not os.path.exists(markup_path):
                print(markup_path)
                sys.exit("No markup file")
        markup = json.load(open(markup_path))
        markup = filter_markup(markup, image_dir)
        markup = {os.path.join(image_dir, key): value for (key, value) in markup.items()}
        union_markup.update(markup)
    return union_markup


def create_lst(union_markup: dict, lst_name: str) -> None:
    print("Creating lst...")
    # Convert json to .lst, format indx\t label_1\t .. label_n\t img_path
    lst_path = lst_name
    with open(lst_path, 'w+') as f:
        for indx, img_path in enumerate(list(union_markup.keys())):
            if type(union_markup[img_path]) == type([]):
                # Full body case ))))
                union_markup[img_path] = [value for value in union_markup[img_path][0]]
                # 2 -- all attributes  # TODO fix this hardcoding
                while len(union_markup[img_path]) != 2:
                    union_markup[img_path].append(-1)
            else:
                # 19 -- all attributes  # TODO fix this hardcoding
                while len(union_markup[img_path]) != 19:
                    union_markup[img_path].append(-1)
            labels = map(str, union_markup[img_path])
            out = f'{indx}\t' + '\t'.join(labels) + '\t' + img_path + '\n'
            f.write(out)


def create_rec(lst_name: str):
    print("Creating rec...")
    os.system(f"python3 {get_original_cwd()}/tools/preproc_data/utils/im2rec.py {lst_name} ./ --pack-label")


def create_df(union_markup: dict, columns: List[str], output_name: str):
    print("Creating df...")
    df = pd.DataFrame(columns=columns)
    for img_path in tqdm(list(union_markup.keys())):
        row = [img_path] + list(union_markup[img_path])[:len(columns)-1]
        df.loc[len(df.index)] = row
        df["Filepath"].apply(str)
    save_path = f'{output_name}.csv'
    df.to_csv(save_path, index=False)


def move_to_data(output_name: str):
    dst = f"{get_original_cwd()}/data/Records"
    if not os.path.exists(dst):
        os.makedirs(dst)
    shutil.move(f"{output_name}.csv", f"{dst}/{output_name}.csv")
    shutil.move(f"{output_name}.rec", f"{dst}/{output_name}.rec")
    shutil.move(f"{output_name}.idx", f"{dst}/{output_name}.idx")


def create_empty_markup(images_dir: List[str]) -> dict:
    union_markup = dict()
    value = [-1]
    for i, image_dir in enumerate(images_dir):
        markup = {key: value for key in os.listdir(image_dir)}
        markup = {os.path.join(image_dir, key): value for (key, value) in markup.items()}
        union_markup.update(markup)
    return union_markup


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_container(cfg)
    images_dir = [dataset['image dir'] for dataset in cfg['datasets']]
    images_dir = [os.path.join(get_original_cwd(), "data", "Datasets", image_dir)
                  for image_dir in images_dir]
    markups = [dataset['markup'] for dataset in cfg['datasets']]
    output_name = cfg['name']
    lst_name = f"{output_name}.lst"

    if cfg['no_label']:
        union_markup = create_empty_markup(images_dir)
    else:
        union_markup = union_markups(images_dir, markups)

    create_lst(union_markup, lst_name)
    create_rec(lst_name)
    create_df(union_markup, cfg['type']['all_attributes'], output_name)
    move_to_data(output_name)

if __name__ == "__main__":
    main()

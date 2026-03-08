import os
import base64
import csv
import json
import random
import shutil
import pandas as pd
from PIL import Image
from io import BytesIO
from natsort import natsorted
from sklearn.model_selection import train_test_split


def rename_files_and_save_to_csv(folder_path='ImageData', csv_file="data_txt/renamed_files.csv"):
    # 函数：将文件按自然数顺序重命名，并返回重命名后的文件名和原始文件名的字典
    def rename_files(folder_path):
        # 获取文件夹中所有文件名，并按自然数顺序排序
        file_names = natsorted(os.listdir(folder_path))

        # 重命名文件并记录原始文件名和重命名后的文件名的对应关系
        renamed_files = {}
        for i, file_name in enumerate(file_names):
            original_name = os.path.join(folder_path, file_name)
            new_name = os.path.join(folder_path, f"{i + 1}{os.path.splitext(file_name)[1]}")
            os.rename(original_name, new_name)
            renamed_files[new_name] = original_name

        return renamed_files

    # 函数：将字典保存到CSV文件中
    def save_dict_to_csv(dictionary, csv_file):
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Renamed Name', 'Original Name'])
            for key, value in dictionary.items():
                writer.writerow([os.path.basename(key), os.path.basename(value)])

    # 重命名文件并获取重命名后的文件名和原始文件名的字典
    renamed_files_dict = rename_files(folder_path)

    # 保存字典到CSV文件
    save_dict_to_csv(renamed_files_dict, csv_file)

    print("Files renamed and dictionary saved to CSV successfully!")


def compress_images(input_folder, output_folder='ImageData', target_resolution=(336, 336)):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 遍历输入文件夹中的所有图像文件
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 遍历输入文件夹中的所有图像文件
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            # 将图像转换为RGB模式（如果尚未处于RGB模式）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # 将图像转换为RGB模式（如果尚未在RGB模式）
            img = img.resize(target_resolution)
            # 将图像转换为RGB模式（如果尚未在RGB模式）
            output_path = os.path.join(output_folder, filename)
            img.save(output_path)
            # print(f"压缩并保存: {filename} -> {output_path}")
    print("compress--finished")

def create_text_ids(csv_file):
    df = pd.read_csv(csv_file, encoding="utf-8")
    df.rename(columns={"caption": "text", 'image_id': 'image_ids'}, inplace=True)
    df['text_id'] = range(1, len(df) + 1)
    df.to_csv(csv_file, index=False)


def split_dataset_with_rename(csv_file, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    # 设置随机种子，以确保每次运行得到的结果一致
    random.seed(seed)

    # 读取.csv文件
    df = pd.read_csv(csv_file)

    # 划分数据集
    train_df, temp_df = train_test_split(df, test_size=1 - train_ratio, random_state=seed)
    val_test_df, test_df = train_test_split(temp_df, test_size=test_ratio / (test_ratio + val_ratio), random_state=seed)

    # 创建文件夹来保存拆分后的数据集
    train_folder = "image_data/train_images"  # 训练集文件夹路径
    val_folder = "image_data/val_images"  # 验证集文件夹路径
    test_folder = "image_data/test_images"  # 测试集文件夹路径
    train_csv = "data_txt/train.csv"
    val_csv = "data_txt/val.csv"
    test_csv = "data_txt/test.csv"
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # 将图片移动到相应的文件夹中
    for index, row in train_df.iterrows():
        image_id = row['image_ids']
        image_path = renamed_files_dict[image_id]
        target_path = os.path.join(train_folder, os.path.basename(image_path))
        shutil.move("ImageData/"+image_path, target_path)

    for index, row in val_test_df.iterrows():
        image_id = row['image_ids']
        image_path = renamed_files_dict[image_id]
        target_path = os.path.join(val_folder, os.path.basename(image_path))
        shutil.move("ImageData/"+image_path, target_path)

    for index, row in test_df.iterrows():
        image_id = row['image_ids']
        image_path = renamed_files_dict[image_id]
        target_path = os.path.join(test_folder, os.path.basename(image_path))
        shutil.move("ImageData/"+image_path, target_path)

    # 保存拆分后的.csv文件
    train_df.to_csv(train_csv, index=False)
    val_test_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    return train_folder, train_csv, val_folder, val_csv, test_folder, test_csv


def create_tsv_file(folder_path, file_path):
    file_names = os.listdir(folder_path)
    jpg_files = [file for file in file_names if file.lower().endswith('.jpg')]
    for jpg_file in jpg_files:
        image_path = os.path.join(folder_path, jpg_file)
    # 将数据写入 .tsv 文件
    with open(file_path, 'w') as tsv_file:
        for jpg_file in jpg_files:
            image_path = os.path.join(folder_path, jpg_file)
            img = Image.open(image_path)
            img_buffer = BytesIO()
            img.save(img_buffer, format=img.format)
            byte_data = img_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")  # 直接进行编码和解码
            # 从文件名中提取图像 ID
            image_id = int(os.path.basename(image_path)[:-4])
            # print(image_id)
            # 将数据写入 .tsv 文件
            tsv_file.write(f"{image_id}\t{base64_str}\n")
    print("create_tsv_file---finished")


# 函数：从 CSV 文件中读取数据

def read_csv_file(csv_file):
    data = []
    with open(csv_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data



# 函数：将转换后的关系写入到 JSONL 文件中
def write_jsonl_file(data, jsonl_file):
    with open(jsonl_file, 'w', encoding='utf-8') as jsonl:
        for row in data:
            jsonl.write(json.dumps(row, ensure_ascii=False) + '\n')


def create_jsonl_file(data_path, jsonl_file):
    # 读取原始 CSV 文件
    data = read_csv_file(data_path)
    # print(type(data))
    data = [{k: int(v) if k == 'text_id' else v for k, v in d.items()} for d in data]

    for row in data:
        original_image_id = row['image_ids']
        if original_image_id in renamed_files_dict:
            row['image_ids'] = [renamed_files_dict[original_image_id][:-4]]
    data = [{k: int(v) if k == 'image_ids' else v for k, v in d.items()} for d in data]
    # 将转换后的数据写入 JSONL 文件
    write_jsonl_file(data, jsonl_file)
    print("JSONL 文件已生成。")


if __name__ == "__main__":
    im_pash = "ImageData_original"
    csv_path = "data_txt/ImageWordData.csv"
    datasets = "datasets/tiandijinghua"
    if not os.path.exists(datasets):
        os.makedirs(datasets)
    compress_images(im_pash)
    rename_files_and_save_to_csv()
    renamed_files_dict = {}
    with open('data_txt/renamed_files.csv', 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过标题行
        for row in reader:
            renamed_files_dict[row[1]] = row[0]
    create_text_ids(csv_path)
    train_folder, train_csv, \
        val_folder, val_csv, \
        test_folder, test_csv = split_dataset_with_rename(csv_path)
    create_tsv_file(train_folder, datasets + "/train_imgs.tsv")
    create_jsonl_file(train_csv, datasets + "/train_texts.jsonl")
    create_tsv_file(val_folder, datasets + "/valid_imgs.tsv")
    create_jsonl_file(val_csv, datasets + "/valid_texts.jsonl")
    create_tsv_file(test_folder, datasets + "/test_imgs.tsv")
    create_jsonl_file("data_txt/test.csv", datasets + "/test_texts.jsonl")

"""
python cn_clip/preprocess/build_lmdb_dataset.py --data_dir datasets/tiandijinghua --splits train,valid,test
"""

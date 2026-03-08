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


def rename_files_and_save_to_csv(folder_path='ImageData', csv_file="data_csv/renamed_files.csv"):
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


def compress_images(input_folder, output_folder, target_resolution=(336, 336)):
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


def create_csv(csv_file):
    df = pd.read_csv(csv_file, encoding="utf-8")
    df['original_text_id'] = df['text_id']
    # 将text_id替换为从1开始的自然数
    df['text_id'] = range(1, len(df) + 1)
    # 将original_text_id和text_id关系保存到单独的CSV文件
    text_id_mapping = df[['original_text_id', 'text_id']]
    text_id_mapping.to_csv('data_csv/res2/text_id_mapping.csv', index=False)
    # 删除original_text_id列
    df.drop(columns=['original_text_id'], inplace=True)
    # 添加一列image_id，值全部为0
    df['image_id'] = 0
    df.rename(columns={"caption": "text", 'image_id': 'image_ids'}, inplace=True)
    df.to_csv(csv_file, index=False)


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
    data = [{k: int(v) if k == 'text_id' else v for k, v in d.items()} for d in data]
    # for row in data:
    #     original_image_id = row['image_ids']
    #     if original_image_id in renamed_files_dict:
    #         row['image_ids'] = [int(renamed_files_dict[original_image_id][:-4])]
    data = [{k: [int(v)] if k == 'image_ids' else v for k, v in d.items()} for d in data]
    # 将转换后的数据写入 JSONL 文件
    write_jsonl_file(data, jsonl_file)
    print("JSONL 文件已生成。")


if __name__ == "__main__":
    """
        处理附件2数据。
    """
    im_pash = "ImageData_text_img"
    csv_path = "data_csv/word_test.csv"
    datasets = "datasets/res1"
    if not os.path.exists(datasets):
        os.makedirs(datasets)
    compress_images(im_pash, 'ImageData_text_img_ys')
    rename_files_and_save_to_csv('ImageData_text_img_ys', "data_csv/res1/renamed_files.csv")
    renamed_files_dict = {}
    with open("data_csv/res1/renamed_files.csv", 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过标题行
        for row in reader:
            renamed_files_dict[row[1]] = row[0]
    create_csv(csv_path)
    create_tsv_file('ImageData_text_img_ys', datasets + "/test_imgs.tsv")
    create_jsonl_file(csv_path, datasets + "/test_texts.jsonl")
    """
        处理附件3数据。
    """
    # im_pash = "ImageData_img_text"
    # csv_path = "data_csv/word_data.csv"
    # datasets = "datasets/res2"
    # if not os.path.exists(datasets):
    #     os.makedirs(datasets)
    # compress_images(im_pash, 'ImageData_img_text_ys')
    # rename_files_and_save_to_csv('ImageData_img_text_ys', "data_csv/res2/renamed_files.csv")
    # renamed_files_dict = {}
    # with open("data_csv/res2/renamed_files.csv", 'r', newline='', encoding='utf-8') as csvfile:
    #     reader = csv.reader(csvfile)
    #     next(reader)  # 跳过标题行
    #     for row in reader:
    #         renamed_files_dict[row[1]] = row[0]
    # create_csv(csv_path)
    # create_tsv_file('ImageData_img_text_ys', datasets + "/test_imgs.tsv")
    # create_jsonl_file(csv_path, datasets + "/test_texts.jsonl")

"""
python cn_clip/preprocess/build_lmdb_dataset.py --data_dir datasets/tiandijinghua --splits train,valid,test
"""

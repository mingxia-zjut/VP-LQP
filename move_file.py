import os
import shutil

folder_to_recurse = 'D:/GCN/25.1.8_minor_revision/公开代码和数据集/VP-LQP-vison-perceptive-link-quality-predictor/dataset/BSS/'
for type in ['test', 'train']:
    if type == 'train':
        folder_path = folder_to_recurse + type + '/' 
        for file_name in os.listdir(folder_path):
            file_path = folder_path + file_name
            print(file_path)
            new_file_path = file_path.replace('30-', '')
            print(new_file_path)
            os.rename(file_path, new_file_path)
        
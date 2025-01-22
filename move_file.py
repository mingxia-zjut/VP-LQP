import os
import shutil

folder_to_recurse = 'D:/GCN/25.1.8_minor_revision/公开代码和数据集/VP-LQP-vison-perceptive-link-quality-predictor/dataset/NDL/test'
for scene in ['indoor', 'outdoor']:
    for num in range(10):
        folder_path = f'{folder_to_recurse}/{scene}/30-{num + 1}'
        for filename in os.listdir(folder_path):
            file_path = folder_path + '/' + filename
            if os.path.isdir(file_path):
                folder_to_delete_path = file_path + '/0.25_0.55'
                print(folder_to_delete_path)        
                os.rmdir(folder_to_delete_path)


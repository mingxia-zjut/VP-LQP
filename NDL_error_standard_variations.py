import numpy as np

anchor_num = 6

# 读入数据
if anchor_num == 6:
    selected_epoch_dict_indoor1 = {'node1': 0, 'node2': 3, 'node3': 9, 'node4': 6, 'node5': 4} 
    selected_epoch_dict_indoor2 = {'node1': 0, 'node2': 2, 'node3': 2, 'node4': 5, 'node5': 3}
    selected_epoch_dict_outdoor1 = {'node1': 2, 'node2': 6, 'node3': 7, 'node4': 1, 'node5': 9}
    selected_epoch_dict_outdoor2 = {'node1': 1, 'node2': 4, 'node3': 6, 'node4': 0, 'node5': 5}
elif anchor_num == 10:
    selected_epoch_dict_indoor1 = {'node1': 7, 'node2': 9, 'node3': 8, 'node4': 0, 'node5': 4} 
    selected_epoch_dict_indoor2 = {'node1': 5, 'node2': 4, 'node3': 0, 'node4': 3, 'node5': 8}
    selected_epoch_dict_outdoor1 = {'node1': 8, 'node2': 1, 'node3': 7, 'node4': 6, 'node5': 1}
    selected_epoch_dict_outdoor2 = {'node1': 4, 'node2': 4, 'node3': 3, 'node4': 7, 'node5': 4}

# all: 所有直线做交点 ran: 随机选取两条直线做交点 our: 我们的置信度方法 max: 选取角度最大的两条直线
methods = ['all', 'ran', 'our', 'max']

all_error, ran_error, our_error, max_error = [], [], [], []

for scene in ['indoor1', 'indoor2', 'outdoor1', 'outdoor2']:
    for node in ['node1', 'node2', 'node3', 'node4', 'node5']:
        selected_epoch = selected_epoch_dict_indoor1[node] if scene == 'indoor1' else selected_epoch_dict_indoor2[node] if scene == 'indoor2' else selected_epoch_dict_outdoor1[node] if scene == 'outdoor1' else selected_epoch_dict_outdoor2[node]
        scene_chinese = '室内定位1' if scene == 'indoor1' else '室内定位2' if scene == 'indoor2' else '室外定位1' if scene == 'outdoor1' else '室外定位2'
        log_file = f"D:/GCN/25.1.8_minor_revision/NDL/result/0.25_0.55_{selected_epoch}_{scene_chinese}_30-{node[-1]}_{anchor_num}.txt"
        print(log_file)
        with open(log_file, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                if line.startswith('our_error'):
                    our_error.append(float(line.split(':')[-1]))
                elif line.startswith('all_error'):
                    all_error.append(float(line.split(':')[-1]))
                elif line.startswith('ran_error'):
                    ran_error.append(float(line.split(':')[-1]))
                elif line.startswith('max_error'):
                    max_error.append(float(line.split(':')[-1]))
        print('our_error.shape:', len(our_error))

all_error, ran_error, our_error, max_error = np.array(all_error), np.array(ran_error), np.array(our_error), np.array(max_error)

# 检查数据
greater_than_95_count = 0
greater_than_90_count = 0
greater_than_85_count = 0
greater_than_80_count = 0
greater_than_75_count = 0
greater_than_70_count = 0
greater_than_65_count = 0
greater_than_60_count = 0
greater_than_55_count = 0
greater_than_50_count = 0
greater_than_45_count = 0
greater_than_40_count = 0
greater_than_35_count = 0
greater_than_30_count = 0
greater_than_25_count = 0
greater_than_20_count = 0
greater_than_15_count = 0
greater_than_10_count = 0
greater_than_5_count = 0
smaller_than_5_count = 0

for error in our_error:
    if error > 0.95:
        greater_than_95_count += 1
    elif error > 0.9:
        greater_than_90_count += 1
    elif error > 0.85:
        greater_than_85_count += 1
    elif error > 0.8:
        greater_than_80_count += 1
    elif error > 0.75:
        greater_than_75_count += 1
    elif error > 0.7:
        greater_than_70_count += 1
    elif error > 0.65:
        greater_than_65_count += 1
    elif error > 0.6:
        greater_than_60_count += 1
    elif error > 0.55:
        greater_than_55_count += 1
    elif error > 0.5:
        greater_than_50_count += 1
    elif error > 0.45:
        greater_than_45_count += 1
    elif error > 0.4:
        greater_than_40_count += 1
    elif error > 0.35:
        greater_than_35_count += 1
    elif error > 0.3:
        greater_than_30_count += 1
    elif error > 0.25:
        greater_than_25_count += 1
    elif error > 0.2:
        greater_than_20_count += 1
    elif error > 0.15:
        greater_than_15_count += 1
    elif error > 0.1:
        greater_than_10_count += 1
    elif error > 0.05:
        greater_than_5_count += 1
    elif error <= 0.05:
        smaller_than_5_count += 1
    
print(f"error大于0.95的有{greater_than_95_count}个")
print(f"error大于0.9的有{greater_than_90_count}个")
print(f"error大于0.85的有{greater_than_85_count}个")
print(f"error大于0.8的有{greater_than_80_count}个")
print(f"error大于0.75的有{greater_than_75_count}个")
print(f"error大于0.7的有{greater_than_70_count}个")
print(f"error大于0.65的有{greater_than_65_count}个")
print(f"error大于0.6的有{greater_than_60_count}个")
print(f"error大于0.55的有{greater_than_55_count}个")
print(f"error大于0.5的有{greater_than_50_count}个")
print(f"error大于0.45的有{greater_than_45_count}个")
print(f"error大于0.4的有{greater_than_40_count}个")
print(f"error大于0.35的有{greater_than_35_count}个")
print(f"error大于0.3的有{greater_than_30_count}个")
print(f"error大于0.25的有{greater_than_25_count}个")
print(f"error大于0.2的有{greater_than_20_count}个")
print(f"error大于0.15的有{greater_than_15_count}个")
print(f"error大于0.1的有{greater_than_10_count}个")
print(f"error大于0.05的有{greater_than_5_count}个")
print(f"error小于等于0.05的有{smaller_than_5_count}个")


# 计算总体标准差
all_std_dev = np.std(all_error)
ran_std_dev = np.std(ran_error)
our_std_dev = np.std(our_error)
max_std_dev = np.std(max_error)

print(f"所有直线做交方法的总体标准差是: {all_std_dev}")
print(f"随机取两条直线做交方法的总体标准差是: {ran_std_dev}")
print(f"置信度最高的两条直线做交的方法的总体标准差是: {our_std_dev}")
print(f"夹角最大的两条直线做交的方法的总体标准差是: {max_std_dev}")

all_mean = np.mean(all_error)
ran_mean = np.mean(ran_error)
our_mean = np.mean(our_error)
max_mean = np.mean(max_error)
print(f"all_mean: {all_mean}\nran_mean: {ran_mean}\nour_mean: {our_mean}\nmax_mean: {max_mean}")



# 计算总体标准差

# std_dev = np.std(data)

# print(f"总体标准差是: {std_dev}")

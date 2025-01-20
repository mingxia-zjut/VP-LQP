# load libraries
import numpy as np
import extention_lib.data_process # 定义了数据处理，计算相关函数
import extention_lib.model        # 定义了置信度网络的模型
import sklearn.linear_model
import torch
import time
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import recall_score, accuracy_score
from sklearn.metrics import precision_score, f1_score
from sklearn.metrics import mean_squared_error
import torch_geometric
# self-defined functions
from extention_lib.data_process import GetClockAngle, get_distance_point2line, nodedistance
from extention_lib.data_process import GeneralEquation, calculation
import torch_geometric.nn as pyg_nn
from extention_lib.model import GAT

print(torch_geometric.__version__)
# deal_data
def deal_data(data):
    # data = [k1, b1, [anchor1_x, anchor1_y, p1], [anchor2_x, anchor2_y, p2], ...]
    x_line = [ii[0] for ii in data[2:][0]]
    y_line = [ii[1] for ii in data[2:][0]]
    xy_line = np.array([x_line, y_line])
    P = [ii[2] * 100 for ii in data[2:][0]]
    center = np.mean(xy_line, axis=1)
    k = data[0]
    b = center[1] - k * center[0]
    v0 = [10, k * (center[0] + 10) + b - center[1]]

    dat = []
    dat.append([0, 0, 0, 100, k])
    dat.append([50, 0, 0, 100, k])
    dat.append([50, 180, 0, 100, k])
    dat.append([50, 90, 50, 100, k])
    dat.append([50, -90, 50, 100, k])
    
    for j in range(len(P)): # len(P)表示锚点数量
        v1 = [x_line[j] - center[0], y_line[j] - center[1]]
        angle = GetClockAngle(v0, v1)
        dis = get_distance_point2line([x_line[j], y_line[j]], [k, b])
        dat.append([nodedistance(np.array([x_line[j], y_line[j]]), center), angle, dis, P[j], k])

    for j in range(len(P) + 5, 19):# 对于锚点数不足14的数据，补 0 做 padding 
        dat.append([0, 0, 0, 0, 0])

    # 制作邻接矩阵
    ad = []

    # 四个虚拟锚点到中心节点双向连接，虚拟锚点之间无连接
    for m in range(len(P) + 5):
        if m == 0:
            ad.append([0, 1, 1, 1, 1] + [0 for n in range(14)])
        elif (m >= 1 and m <= 4):
            ad.append([1] + [0 for n in range(18)])
        else:
            ad.append([1, 1, 1, 1, 1] + [0 for n in range(14)])

    for j in range(len(P) + 5, 19):
        ad.append([0 for n in range(19)])
    return dat, ad

line_number = 4
TIME_STEP = 10
epochs = 1
scenes = ['室内定位1', '室内定位2', '室外定位1', '室外定位2']
anchor_num_list = [6]
# scenes = ['室内定位1']
# anchor_num_list = [2,4,6,8,10]
pointpoints = [0, 1, 2, 3, 4]

time_list = []
all_test_error, ran_test_error, max_test_error, our_test_error = [], [], [], []
count = 0
for cishu in range(10):
    canshu = '0.25_0.55_' + str(cishu)
    for scene in scenes:
        for pointpoint in pointpoints:
            # 测试不同锚点对指定节点的定位性能 影响
            for anchor_num in anchor_num_list:
                
                # 加载真实链路，其中真实链路由直线的斜率 k 和 纵截距 b 表示
                t_line = np.load('/home/lzy/2025.1.8_first_revision/LFRE_latency/data/data_for_LFRE_test/' + str(scene) + '/30-' + str(pointpoint + 1) + '/true_line.npy', allow_pickle=True)

                print('scene:', scene, 'pointpoint:', pointpoint, 'anchor_num:', anchor_num, 'canshu:', canshu)
                all_test_error, ran_test_error, max_test_error, our_test_error = [], [], [], []
                final_line = []
                FF = []
                Y = []

                for number in range(line_number):
                    anchor_set_data_file = f'/home/lzy/2025.1.8_first_revision/LFRE_latency/data/data_for_LFRE_test/{scene}/30-{str(pointpoint+1)}/{str(anchor_num)}/{str(canshu)}/{scene}_30-{str(pointpoint+1)}_{str(anchor_num)}_{str(number+1)}.npy'

                    # anchor_set.shape: (100, anchor_num, 3), anchor_set[0][0] = [anchor_x, anchor_y, anchor_confidence], 其中 anchor_confidence 为由射频遮挡模块预测的直线度量
                    # 随机重复了 100 次的锚点选择, 用于测试不同的锚点集对直线拟合的影响， 所以这里 anchor_set 的长度为 100
                    anchor_set = np.load(anchor_set_data_file, allow_pickle=True)

                    # 真实直线
                    line_true = t_line[number]
                    
                    # 通过锚点对直线进行拟合
                    clf = sklearn.linear_model.Ridge(alpha=0.5)
                    
                    for m in range(100):
                        l2 = anchor_set[m]
                        
                        x1, y1, z1 = [], [], []
                        for j in range(len(l2)):
                            x1.append(l2[j][0])
                            y1.append(l2[j][1])
                            z1.append(l2[j][2])

                        # 通过锚点集 拟合直线， 如果 只有两个锚点， 则可以直接带公式计算直线的参数 k, b。否则， 通过线性拟合确定直线
                        if len(x1) == 2:
                            k1, b1 = GeneralEquation(x1[0], y1[0], x1[1], y1[1])
                            line_test = [k1, b1]
                            line_data = [k1, b1, l2.tolist()]
                        else:
                            pointx = x1
                            pointy = y1
###############################################################################################################################
############################################# 以下代码参考论文第三章-链路拟合模块的介绍 ###########################################
                            px = np.array(pointx).reshape(-1, 1)
                            py = np.array(pointy)

                            clf.fit(px, py)
                            y_pred = clf.predict(px)

                            MSN1 = mean_squared_error(y_pred, py)

                            for u in range(len(y_pred)):
                                if (px[u] != px[-1]):
                                    break

                            k1, b1 = GeneralEquation(px.reshape(1, -1).squeeze()[u], y_pred[u],
                                                     px.reshape(1, -1).squeeze()[-1], y_pred[-1])

                            px = np.array([-i for i in pointy]).reshape(-1, 1)
                            py = np.array([-i for i in pointx])
                            
                            clf.fit(px, py)
                            y_pred = clf.predict(px)
                            
                            MSN2 = mean_squared_error(y_pred, py)
                            
                            p_y = np.array([-i for i in px.reshape(1, -1).squeeze()])
                            p_x = np.array([-i for i in py])
                            
                            ppx = np.array([-i for i in y_pred])
                            
                            for n in range(len(p_y)):
                                if (ppx[n] != ppx[-1]):
                                    break
                            
                            k2, b2 = GeneralEquation(ppx.reshape(1, -1).squeeze()[n], p_y[n],
                                                     ppx.reshape(1, -1).squeeze()[-1], p_y[-1])
###############################################################################################################################                            
                            if MSN1 < MSN2:
                                line_test = [k1, b1]
                                line_data = [k1, b1, l2.tolist()]
                            else:
                                line_test = [k2, b2]
                                line_data = [k2, b2, l2.tolist()]

                        if calculation(line_test, line_true) <= 0.55:
                            Y.append(-1)
                        else:
                            Y.append(1)
                        
                        # 通过置信度网络做预测，评估模型性能
                        path = '/home/lzy/2025.1.8_first_revision/LFRE_latency/trained_model/model_test11.pkl'
                        device = torch.device('cuda')
                        data, adj = deal_data(line_data)
                        data = torch.tensor(data, dtype=torch.float).unsqueeze(0).to(device)
                        adj = torch.tensor(adj, dtype=torch.float).unsqueeze(0).to(device)

                        # data = torch.tensor(data, dtype=torch.float).unsqueeze(0)
                        # adj = torch.tensor(adj, dtype=torch.float).unsqueeze(0)

                        new_model = GAT(5, 2).to(device) 
                        new_model.load_state_dict(torch.load(path))
                        
                        # new_model = torch.load(path)
                        new_model.eval()
                        mask = [[True] + [False for i in range(18)]]
                        mask = torch.tensor(mask, dtype=torch.bool).to(device)
                        
                        time_start = time.clock()
                        output, forward_time = new_model(data, adj)
                        time_end = time.clock()

                        time_list.append(forward_time)
                        # print('time:', time_end - time_start)
                        LL = list(torch.max(output[mask].cpu(), 1)[1].data.numpy())[0]
                        if LL == 0:
                            LL = -1
                        FF.append(LL)
                target = np.array(Y)
                preds = np.array(FF)
                acc = accuracy_score(target, preds)
                pre = precision_score(target, preds)
                rec = recall_score(target, preds)
                fpr, tpr, thresholds_keras = roc_curve(target, preds)
                auc_score = auc(fpr, tpr)
                print(f'Accuracy on : ', acc)
                print(f'Pre on : ', pre)
                print(f'Rec on : ', rec)
                print(f'FPR on : ', fpr)
                print(f'TPR on : ', tpr)
                print(f'AUC on : ', auc_score)
                string_list = ''
                string_list += str(canshu) + '_' + str(scene) + '_30-' + str(pointpoint + 1) + '_' + str(anchor_num) + ':\n'
                string_list += 'Accuracy on : ' + str(acc) + '\n'
                string_list += 'Pre on : ' + str(pre) + '\n'
                string_list += 'Rec on : ' + str(rec) + '\n'
                string_list += 'FPR on : ' + str(fpr) + '\n'
                string_list += 'TPR on : ' + str(tpr) + '\n'
                string_list += 'AUC on : ' + str(auc_score) + '\n'
                
                f = open('/home/lzy/2025.1.8_first_revision/LFRE_latency/trained_model/model_1_with_time_测试结果.txt', 'a', encoding='Utf-8')
                f.write(string_list)
                f.close()

                print()
f = open('/home/lzy/2025.1.8_first_revision/LFRE_latency/trained_model/model_1_with_time_new_测试结果.txt', 'a', encoding='Utf-8')
f.write(f'mean_lfre_time: {str(np.mean(time_list))}')
f.close()
print('time:', np.mean(time_list))
print('测试结束')
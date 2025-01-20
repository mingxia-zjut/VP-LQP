from openpyxl import load_workbook
import torch
from scipy import fft
import numpy as np
import matplotlib.pyplot as plt
from math import pow
import random
from tsai.all import *

import torch.optim as optim
from torch.autograd import Variable
import math
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import mean_squared_error
import sklearn.linear_model
from torch.nn.parameter import Parameter
import torch_geometric.nn as pyg_nn
from torch_geometric.datasets import Planetoid
import scipy.sparse as sp

from data_preprocess import GetClockAngle, get_distance_point2line, nodedistance,  panduan, nodedistance, GeneralEquation, calculation
from model import GAT

def deal_data_without_label(da): # 在测试时是直接对测试数据计算置信度得到标签的， 所以这里没有标签，训练时 deal_data是要读标签的，因为标签已经做在训练数据里了
    x_line = [ii[0] for ii in da[2:][0]]
    y_line = [ii[1] for ii in da[2:][0]]
    xy_line = np.array([x_line, y_line])
    P = [ii[2] * 100 for ii in da[2:][0]]
    center = np.mean(xy_line, axis=1)
    k = da[0]
    b = center[1] - k * center[0]
    v0 = [10, k * (center[0] + 10) + b - center[1]]

    dat = []
    dat.append([0, 0, 0, 100, k])
    dat.append([50, 0, 0, 100, k])
    dat.append([50, 180, 0, 100, k])
    dat.append([50, 90, 50, 100, k])
    dat.append([50, -90, 50, 100, k])
    
    for j in range(len(P)):
        v1 = [x_line[j] - center[0], y_line[j] - center[1]]
        angle = GetClockAngle(v0, v1)
        dis = get_distance_point2line([x_line[j], y_line[j]], [k, b])
        dat.append([nodedistance(np.array([x_line[j], y_line[j]]), center), angle, dis, P[j], k])
    for j in range(len(P) + 5, 19):
        dat.append([0, 0, 0, 0, 0])

    ad = []
###################################################################################################################################################### 
############################      这里因为置信度网络测试下来第一种情况最好，所以选择第一种情况来做定位               ########################################
    # （1） 四个虚拟锚点到中心节点双向连接，虚拟锚点之间无连接
    for m in range(len(P) + 5):
        if m == 0:
            ad.append([0, 1, 1, 1, 1] + [0 for n in range(14)])
        elif (m >= 1 and m <= 4):
            ad.append([1] + [0 for n in range(18)])
        else:
            ad.append([1, 1, 1, 1, 1] + [0 for n in range(14)])
######################################################################################################################################################
    for j in range(len(P) + 5, 19):
        ad.append([0 for n in range(19)])

    return dat, ad



random.seed(1)
line_number = 4
TIME_STEP = 10
scenes = ['室内定位1','室内定位2','室外定位1','室外定位2']
# scenes = ['室内定位2']
maopoint = [2,4,6,8,10]
# maopoint = [6]
pointpoints = [0,1,2,3,4]
# pointpoints = [4]
canshu = '0.25_0.55'

true_pp = [
             [[290,-220],[649,-216],[376,-16],[74,-52],[123,-461]], # 室内定位 1 节点位置
             [[265,-311],[652,-254],[487,-49],[230,-31],[-20,-141]],# 室内定位 2 节点位置
             [[318,-207],[603,-148],[322,-48],[-5,-145],[142,-448]],# 室外定位 1 节点位置
             [[386,-316],[654,-193],[463,-80],[233,-87],[-10,-165]] # 室外定位 2 节点位置
            ]
for mao_point in maopoint:
    for scene in scenes:
        for pointpoint in pointpoints:
            true_point = true_pp[scenes.index(scene)][pointpoint]
            for cishu in range(10):
                canshu = '0.25_0.55_'+str(cishu)
                save_file = str(canshu)+'_'+str(scene)+'_30-'+str(pointpoint+1)+'_'+str(mao_point)+'.txt'
                t_line = np.load('D:\\GCN\\graphnet\\data\\'+str(scene)+'\\30-'+str(pointpoint+1)+'\\true_line.npy' , allow_pickle=True)
                string_list = ''
                all_string_list = ''
                all_test_error,ran_test_error,max_test_error,our_test_error = [],[],[],[]
                # FFF = []
                for m in range(100):
                    print('epochs: ',m)
                    string_list+='epochs: '+str(m)+'\n'
                    final_line = []
                    FF=[]
                    for number in range(line_number):
                        line_true = t_line[number]
                        data_file ='D:\\GCN\\graphnet\\data\\'+str(scene)+'\\30-'+str(pointpoint + 1)+'\\'+str(mao_point)+'\\'+str(canshu)+'\\'+str(scene)+'_30-'+str(pointpoint+1)+'_'+str(mao_point)+'_'+str(number+1)+'.npy'
                        
                        X = np.load(data_file, allow_pickle=True)
                        l2 = X[m]
                        
                        clf=sklearn.linear_model.Ridge(alpha=0.5)
                        x1,y1,z1=[],[],[]
                        for j in range(len(l2)):
                            x1.append(l2[j][0])
                            y1.append(l2[j][1])
                            z1.append(l2[j][2])
                        string_list +='pre_point:\n'
                        string_list +='x:'+str(x1)+'\n'
                        string_list +='y:'+str(y1)+'\n'
                        string_list +='z:'+str(z1)+'\n'
                        
                        #锚点拟合成直线
                        if len(x1)==2:
                            k1,b1 = GeneralEquation(x1[0],y1[0],x1[1],y1[1])
                            line_data = [k1,b1,l2.tolist()]
                            line_test=[k1,b1]
                        else:
                            pointx = x1
                            pointy = y1          
                            # 导入数据集
                            px = np.array(pointx).reshape(-1,1)
                            # px = np.array(pointx)
                            py = np.array(pointy)
                            # 训练模型
                            clf.fit(px,py)
                            # 测试模型
                            y_pred=clf.predict(px)
                            MSN1 =mean_squared_error(y_pred,py)
                            for u in range(len(y_pred)):
                                if (px[u]!=px[-1]):
                                    break
                            k1,b1 = GeneralEquation(px.reshape(1,-1).squeeze()[u],y_pred[u],px.reshape(1,-1).squeeze()[-1],y_pred[-1])

                            px = np.array([-i for i in pointy]).reshape(-1,1)
                            py = np.array([-i for i in pointx])
                            clf.fit(px,py)
                            y_pred=clf.predict(px)
                            MSN2 = mean_squared_error(y_pred,py)

                            p_y = np.array([-i for i in px.reshape(1,-1).squeeze()])
                            p_x = np.array([-i for i in py])
                            ppx = np.array([-i for i in y_pred])
                            for n in range(len(p_y)):
                                if (ppx[n]!=ppx[-1]):
                                    break
                            k2,b2 = GeneralEquation(ppx.reshape(1,-1).squeeze()[n],p_y[n],ppx.reshape(1,-1).squeeze()[-1],p_y[-1])
                            if MSN1<MSN2:
                                line_data = [k1,b1,l2.tolist()]
                                line_test=[k1,b1]
                            else:
                                line_data = [k2,b2,l2.tolist()]
                                line_test=[k2,b2]

                        if calculation(line_test,line_true)<=0.55:
                            FF.append(-1)
                            # FFF.append(-1)
                        else:
                            FF.append(1)
                            # FFF.append(1)
                        #获取拟合直线置信度
                        data,adj = deal_data_without_label(line_data)
                        data = torch.tensor(data,dtype=torch.float).unsqueeze(0)
                        adj = torch.tensor(adj,dtype=torch.float).unsqueeze(0)
                        path = 'E:\\link_quality_estimate\\Initialization_Phase_NDL\\Cross_Modality_localization\\置信度网络\\trained_model\\model_1.pkl'
                        # path = 'D:\\GCN\\定位校准\\Confidence_Net_models\\0.8拟合修改数据修改邻接矩阵全连接_加4_5锚点_2分类_0.55_0.9340_0.9351_0.9775\\model.pkl'
                        new_model = torch.load(path)
                        mask = [[True]+[False for i in range(18)]]
                        mask =torch.tensor(mask,dtype=torch.bool)
                        output = new_model(data, adj)   
                        LL = list(torch.max(output[mask], 1)[1].data.numpy())[0] 
                        if LL==0:
                            LL=0.5
                        elif LL==1:
                            LL=1
                        #含置信度的直线表示
                        final_line.append([line_data[0],line_data[1],LL])
                    print('pre_line:',final_line)  
                    print(FF)
                    string_list += 'pre_line:' + str(final_line) + '\n'
                    string_list += str(FF) + '\n'

                    max_best_FF,our_best_FF = 0,0 
                    max_best_i,max_best_j = 0,1
                    our_best_i,our_best_j = 0,1
                    #求可行度F
                    for i in range(len(final_line)-1):
                        for j in range(i + 1, len(final_line)):
                            our_FF = (np.arctan(abs((final_line[i][0]-final_line[j][0])/(1+final_line[i][0]*final_line[j][0]))))*(final_line[i][2]+final_line[j][2])
                            print('i: ', i, 'j: ', j)
                            print(our_FF)
                            max_FF = np.arctan((abs((final_line[i][0]-final_line[j][0])/(1+final_line[i][0]*final_line[j][0]))))
                            if max_FF > max_best_FF:
                                max_best_FF = max_FF
                                max_best_i = i
                                max_best_j = j
                            if our_FF > our_best_FF:
                                our_best_FF = our_FF
                                our_best_i = i
                                our_best_j = j

                    #置信度方法：
                    point_x,point_y = 0,0
                    if final_line[our_best_i][0]-final_line[our_best_j][0]==0:
                        point_x = (final_line[our_best_j][1]-final_line[our_best_i][1])/(final_line[our_best_i][0]-final_line[our_best_j][0]+0.001)
                        point_y = (final_line[our_best_i][0]*point_x)+final_line[our_best_i][1]
                    else:
                        point_x = (final_line[our_best_j][1]-final_line[our_best_i][1])/(final_line[our_best_i][0]-final_line[our_best_j][0])
                        point_y = (final_line[our_best_i][0]*point_x)+final_line[our_best_i][1]
                    print('our_best_i:',our_best_i)
                    print('our_best_j:',our_best_j)
                    print('our_pre_point:',[point_x,point_y])
                    our_err = panduan([point_x,point_y],true_point)
                    print('our_error:',our_err)
                    our_test_error.append(our_err)
                    string_list += 'our_best_i:'+str(our_best_i)+'\n'
                    string_list += 'our_best_j:'+str(our_best_j)+'\n'
                    string_list += 'our_pre_point:'+str([point_x,point_y])+'\n'
                    string_list += 'our_error:'+str(our_err)+'\n'

                    #夹角最大方法：
                    point_x, point_y = 0, 0
                    if final_line[max_best_i][0]-final_line[max_best_j][0]==0:
                        point_x = (final_line[max_best_j][1]-final_line[max_best_i][1])/(final_line[max_best_i][0]-final_line[max_best_j][0]+0.001)
                        point_y = (final_line[max_best_i][0]*point_x)+final_line[max_best_i][1]
                    else:
                        point_x = (final_line[max_best_j][1]-final_line[max_best_i][1])/(final_line[max_best_i][0]-final_line[max_best_j][0])
                        point_y = (final_line[max_best_i][0]*point_x)+final_line[max_best_i][1]
                    print('max_best_i:',max_best_i)
                    print('max_best_j:',max_best_j)
                    print('max_pre_point:',[point_x,point_y])
                    max_err = panduan([point_x,point_y],true_point)
                    print('max_error:',max_err)
                    max_test_error.append(max_err)
                    string_list +='max_best_i:'+str(max_best_i)+'\n'
                    string_list +='max_best_j:'+str(max_best_j)+'\n'
                    string_list +='max_pre_point:'+str([point_x,point_y])+'\n'
                    string_list +='max_error:'+str(max_err)+'\n'

                    #取随机
                    best = random.sample(range(0,4),2)
                    ran_best_i,ran_best_j = best[0],best[1]
                    point_x,point_y = 0,0
                    if final_line[ran_best_i][0]-final_line[ran_best_j][0]==0:
                        point_x = (final_line[ran_best_j][1]-final_line[ran_best_i][1])/(final_line[ran_best_i][0]-final_line[ran_best_j][0]+0.001)
                        point_y = (final_line[ran_best_i][0]*point_x)+final_line[ran_best_i][1]
                    else:
                        point_x = (final_line[ran_best_j][1]-final_line[ran_best_i][1])/(final_line[ran_best_i][0]-final_line[ran_best_j][0])
                        point_y = (final_line[ran_best_i][0]*point_x)+final_line[ran_best_i][1]
                    print('ran_best_i:',ran_best_i)
                    print('ran_best_j:',ran_best_j)
                    print('ran_pre_point:',[point_x,point_y])
                    ran_err = panduan([point_x,point_y],true_point)
                    print('ran_error:',ran_err)
                    ran_test_error.append(ran_err)
                    string_list +='ran_best_i:'+str(ran_best_i)+'\n'
                    string_list +='ran_best_j:'+str(ran_best_j)+'\n'
                    string_list +='ran_pre_point:'+str([point_x,point_y])+'\n'
                    string_list +='ran_error:'+str(ran_err)+'\n'

                    #所有直线做交
                    x_line,y_line,xy_line = [],[],[]
                    point_x,point_y = 0,0
                    for i in range(len(final_line)-1):
                        for j in range(i+1,len(final_line)):
                            if final_line[i][0]-final_line[j][0]==0:
                                point_x = (final_line[j][1]-final_line[i][1])/(final_line[i][0]-final_line[j][0]+0.001)
                                point_y = (final_line[i][0]*point_x)+final_line[i][1]
                            else:
                                point_x = (final_line[j][1]-final_line[i][1])/(final_line[i][0]-final_line[j][0])
                                point_y = (final_line[i][0]*point_x)+final_line[i][1]
                            x_line.append(point_x)
                            y_line.append(point_y)
                    xy_line = np.array([x_line,y_line])
                    center = np.mean(xy_line,axis=1)
                    print('all_pre_point:',center)
                    all_err = panduan(center,true_point)
                    print('all_error:',all_err)
                    all_test_error.append(all_err)     
                    string_list +='all_pre_point:'+str(center)+'\n'
                    string_list +='all_error:'+str(all_err)+'\n'

                #总误差输出
                all_mean_error = np.mean(all_test_error)
                ran_mean_error = np.mean(ran_test_error)
                max_mean_error = np.mean(max_test_error)
                our_mean_error = np.mean(our_test_error)
                print('all_mean_error:', all_mean_error)
                print('ran_mean_error:', ran_mean_error)
                print('max_mean_error:', max_mean_error)
                print('our_mean_error:', our_mean_error)
                string_list += '++++++++++++++++++++\n'
                string_list += 'all_mean_error:' + str(all_mean_error) + '\n'
                string_list += 'ran_mean_error:' + str(ran_mean_error) + '\n'
                string_list += 'max_mean_error:' + str(max_mean_error) + '\n'
                string_list += 'our_mean_error:' + str(our_mean_error) + '\n'
                f=open('D:\\GCN\\25.1.8_minor_revision\\NDL\\test5\\'+save_file,'w')
                f.write(string_list)
                f.close() 

                all_string_list += '++++++++场景:' + scene + '第' + str(cishu) + '轮次,' + str(mao_point) + '锚点对节点' + str(pointpoint + 1) + '的定位总误差++++++++++++\n'
                all_string_list += 'all_mean_error:'+str(all_mean_error)+'\n'
                all_string_list += 'ran_mean_error:'+str(ran_mean_error)+'\n'
                all_string_list += 'max_mean_error:'+str(max_mean_error)+'\n'
                all_string_list += 'our_mean_error:'+str(our_mean_error)+'\n'
                f=open('D:\\GCN\\25.1.8_minor_revision\\NDL\\test5\\0.25_0.55总误差.txt','a')
                f.write(all_string_list)
                f.close() 


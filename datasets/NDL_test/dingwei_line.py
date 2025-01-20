import xlrd
import xlwt  
import openpyxl 
from openpyxl import load_workbook 
import torch
from scipy import fft
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from math import pow
import random
import tsai
from tsai.all import *
from tsai.inference import load_learner
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import recall_score,accuracy_score
from sklearn.metrics import precision_score,f1_score
import time
import copy
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


def GeneralEquation(first_x,first_y,second_x,second_y):
    # 一般式 Ax+By+C=0
    A = second_y-first_y
    B = first_x-second_x
    if B==0:
        B = 1
    C = second_x*first_y-first_x*second_y
    k = -1 * A / B
    b = -1 * C / B
    return k, b

# true_pp = [
#     [[290, -220], [649, -216], [376, -16], [74, -52], [123, -461]], # 室内 1 中各个节点的真实位置
#     [[265, -311], [652, -254], [487, -49], [230, -31], [-20, -141]],# 室内 2 中各个节点的真实位置
#     [[318, -207], [603, -148], [322, -48], [-5, -145], [142, -448]],# 室外 1 中各个节点的真实位置
#     [[386, -316], [654, -193], [463, -80], [233, -87], [-10, -165]] # 室外 2 中各个节点的真实位置
# ]

if __name__ == '__main__':
    k1,b1 = GeneralEquation(-10,-165,654,-193)
    k2,b2 = GeneralEquation(-10,-165,463,-80)
    k3,b3  = GeneralEquation(-10,-165,386,-316)
    k4,b4  = GeneralEquation(-10,-165,233,-87)
    true_line = np.array([[k1,b1],[k2,b2],[k3,b3],[k4,b4]])
    np.save('D:\\GCN\\graphnet\\data\\室外定位2\\30-5\\true_line.npy',true_line)
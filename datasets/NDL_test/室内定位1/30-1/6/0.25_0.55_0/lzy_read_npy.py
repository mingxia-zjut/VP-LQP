import numpy as np
import sys

npyFileName = '室内定位1_30-1_6_1.npy'
currentPath = sys.path[0]

data = np.load(currentPath + '\\' + npyFileName, allow_pickle = True)  #加载文件

print('data shape:', data.shape)
print('data type', type(data))

doc = open(currentPath + '\\' +'室内定位1_30-1_6_1.txt', 'w')  #打开一个存储文件，并依次写入
print(npyFileName, file = doc)
for x in data:
    print(x, file=doc)  #将打印内容写入文件中

doc.close()
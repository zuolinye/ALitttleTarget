# -*- coding: utf-8 -*-
import csv

f = open('../results/beginner05.csv', 'r')
content = csv.reader(f)
lineNum = 0

for line in content:
    lineNum += 1
    print(lineNum)  # lineNum就是你要的文件行数
f.close()

import pandas as pd

# data = pd.read_csv('../results/beginner05.csv',nrows =5)
data = pd.read_csv('../results/beginner05.csv')
data2 =pd.read_csv('E:/work/Train/9-2.csv')
print(data)
print(data2)

f.close()



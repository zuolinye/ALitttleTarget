# -*- coding: utf-8 -*-
# 换行
import pandas as pd

# # data = pd.read_csv('../results/beginner05.csv',nrows =5)
# data = pd.read_csv('../results/beginner05.csv')
#
# data2 =pd.read_csv('E:/work/Train/9-2.csv')
# print(data)
# print(data2)
#
# id=data.pop('id')
# data.insert(0,'id',id)
# print (data)
# data.to_csv('../results/changeCol.csv', index = False)
#

data3 = pd.read_csv('../results/changeCol.csv')

print(data3)

data = pd.read_csv('../results/data_lda.pkl_sklearn_svm.csv')
id=data.pop('id')
data.insert(0,'id',id)
print (data)
data.to_csv('../results/data_lda.pkl_sklearn_svm1.csv', index = False)





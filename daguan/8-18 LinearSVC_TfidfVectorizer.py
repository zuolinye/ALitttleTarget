# 导包

import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
start_time = time.time()

"""
代码功能简介：从硬盘上读取已下载好的数据，并进行简单处理
知识点定位：数据预处理
"""
df_train = pd.read_csv('./train_set.csv') # 读取训练集数据
df_test = pd.read_csv('./test_set.csv') # 读取测试集数据
df_train.drop(columns = ['article','id'], inplace = True) # 删除测试集中的article列
df_test.drop(columns = ['article'],inplace = True) # 删除测试集中的article列

"""
代码功能简介：将数据集中的字符文本转换成数字向量，以便计算机能够进行处理（一段文字 --> 一个向量）
知识点定位：特征工程
"""
vectorizer = TfidfVectorizer(min_df = 6, max_df = 0.9, max_features = 100000) # 初始化一个TfidfVectorizer对象

x_train = vectorizer.fit_transform(df_train['word_seg']) # 将一篇文章转为与其对应的一个特征向量
x_test = vectorizer.transform(df_test['word_seg']) # 将一篇文章转为与其对应的一个特征向量
y_train = df_train['class'] - 1 # 因为从0开始计算，所以要将原值-1

"""
代码功能简介：训练一个分类器
知识点定位：传统监督学习算法之对数几率回归（逻辑回归）
"""
classifier = svm.LinearSVC() # 初始化一个分类器
classifier.fit(x_train, y_train) # 训练这个分类器

# 根据上面训练好的分类器对测试集的每个样本进行预测
y_test = classifier.predict(x_test)

# 将测试集的预测结果保存至本地
df_test['class'] = y_test.tolist() # 转换为Python的List形式
df_test['class'] = df_test['class'] + 1 # 将class + 1，保证和官方的预测值一致
df_result = df_test.loc[:, ['id', 'class']]
df_result.to_csv('./result.csv', index = False) # 将结果保存至本地文件

end_time = time.time()
duration = end_time - start_time
print(duration)
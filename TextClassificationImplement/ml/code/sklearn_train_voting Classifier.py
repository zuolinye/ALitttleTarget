# -*- coding: utf-8 -*-
"""
集成了三个不同的分类器（逻辑回归、支持向量机、决策树），用soft voting进行集成,
对机器学习模型进行训练，并对测试集进行预测，并将结果保存至本地
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import time
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier



t_start = time.time()

"""是否开启验证集模式"""
status_vali = True

"""=====================================================================================================================

1 读取数据
"""
data_fp = open('../features/data_tfidf.pkl', 'rb')
x_train, y_train, x_test = pickle.load(data_fp)
data_fp.close()

"""划分训练集和验证集，验证集比例为test_size"""
if status_vali:
    x_train, x_vali, y_train, y_vali = train_test_split(x_train, y_train, test_size=0.1, random_state=0)

"""=====================================================================================================================
print ('-[stage 2]  ...')
2 训练分类器
集成学习voting Classifier在sklearn中的实现
"""
voting_clf = VotingClassifier(estimators = [
    ('log_clf',LogisticRegression(penalty='l2',C = 4, dual = True)),
    ('svm_clf', svm.SVC()),
    ('dt_clf', DecisionTreeClassifier(random_state=666))],
 voting = 'soft')

voting_clf.fit(x_train,y_train)

# """=====================================================================================================================
# print ('-[stage Ⅲ]  ...')
# # 3 在验证集上评估模型
# """
# # if status_vali:
# #     pre_vali = voting_clf.predict(x_vali)
# #     score_vali = f1_score(y_true=y_vali, y_pred=pre_vali, average='macro')
# #     print("验证集分数：{}".format(score_vali))
# #
# # """=====================================================================================================================
# # print ('-[stage Ⅳ]  ...')
# # 4 对测试集进行预测;将预测结果转换为官方标准格式；并将结果保存至本地
# """
y_test = voting_clf.predict(x_test) + 1

df_result = pd.DataFrame(data={'id':range(102277), 'class': y_test.tolist()})
result_path = '../results/' + '_sklearn_' + 'voting Classifier' + '.csv'
df_result.to_csv(result_path, index=False)

t_end = time.time()
print("训练结束，耗时:{}min".format((t_end - t_start) / 60))

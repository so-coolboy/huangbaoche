# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 22:39:21 2017

@author: www
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from collections import Counter

from xgboost import plot_importance
from sklearn.model_selection import GridSearchCV

from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler


#读入提取的特征
feature1 = pd.read_csv(r'E:\data\黄包车比赛\feature101.csv',encoding='gbk')
feature1 = pd.read_csv(r'E:\data\黄包车比赛\feature16.csv',encoding='gbk')

feature = pd.merge(feature1, feature, how='left', on='userid')
feature = pd.read_csv(r'E:\data\黄包车比赛\feature46.csv')
#feature = pd.read_csv(r'E:\data\黄包车比赛\feature90.csv')
feature.fillna("缺失", inplace=True)
feature['HisorderType'].replace('缺失',3,inplace=True)
feature['age'].replace({'缺失':0,'80后':1,'90后':2,'70后':3,'60后':4,'00后':5},inplace=True)
feature.drop('province',axis=1,inplace=True)
#读入标签
orderFuture_train = pd.read_csv(r'E:\data\黄包车比赛\皇包车比赛\皇包车比赛数据-非压缩包\trainingset\orderFuture_train.csv')
orderFuture_test = pd.read_csv(r'E:\data\黄包车比赛\皇包车比赛\皇包车比赛数据-非压缩包\test\orderFuture_test.csv')


#拆分训练集和测试集
train = pd.merge(orderFuture_train, feature, how='left', on='userid')
test = pd.merge(orderFuture_test, feature, how='left', on='userid')

#删去训练集id， 保存测试集id
train.drop('userid', axis=1, inplace=True)

userid = test['userid'].values
test.drop('userid', axis=1, inplace=True)

#划分训练集和验证集
X_train, X_test, y_train, y_test = train_test_split(train[train.columns[1:]],
          train[train.columns[:1]],test_size = 0.25, random_state = 100)

#过采样
#1.使用smote， 准确率是83%
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_sample(X_train, y_train)

#2.使用朴素随机过采样    准确率上升了，现在达到了86%
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_sample(X_train, y_train)


X_resampled = pd.DataFrame(X_resampled, columns = X_train.columns)
y_resampled = pd.DataFrame(y_resampled, columns = ['orderType'])

#3.使用adasyn，生成数据速度十分慢，准确率低，只有81%
from imblearn.over_sampling import ADASYN
X_resampled, y_resampled = ADASYN().fit_sample(X_train, y_train)

#4.使用smote的变体，准确率维持在83%
X_resampled, y_resampled = SMOTE(kind='borderline1').fit_sample(X_train, y_train)


#下采样  
#1.原型生成(prototype generation)
from imblearn.under_sampling import ClusterCentroids

cc = ClusterCentroids(random_state=0)
X_resampled, y_resampled = cc.fit_sample(X_train, y_train)  





#使用xgboost进行训练
#1.设置参数
params = {
    'learning_rate':0.06,
    'n_estimators':1000,
    'max_depth':5,
    'min_child_weight':1,
    'gamma':0,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'objective':'binary:logistic',
    #'alpha':0.2,
    #在各类别样本十分不平衡时，把这个参数设定为正值，可以是算法更快收敛
    #含义：二分类中正负样本比例失衡，需要设置正样本的权重。比如当正负样本比例为1:10时，可以调节scale_pos_weight=10。
    'scale_pos_weight':1
}

clf = xgb.XGBClassifier(**params)
clf.fit(X_resampled, y_resampled,eval_metric='auc', 
         verbose=True, eval_set=[(X_test, y_test.values)], 
                                 early_stopping_rounds=100)
clf.fit(X_train, y_train,eval_metric='auc', 
         verbose=True, eval_set=[(X_test, y_test.values)], 
                                 early_stopping_rounds=100)
#2.对验证集进行预测
preds_proba = clf.predict_proba(X_test)


accuracy = roc_auc_score(y_test, preds_proba[:,1:])     95.43%
print("Accuracy: %.2f%%" % (accuracy * 100.0))    


#3.对测试集进行预测
test_preds_proba = clf.predict_proba(test)
a = pd.DataFrame(test_preds_proba, columns=['as','asd'])
data = {'userid':userid, 'orderType':a['asd'].values}
submit = pd.DataFrame(data, columns=['userid', 'orderType'])
submit.to_csv(r'E:\data\黄包车比赛\皇包车比赛\submit44.csv', index=False)  #准确率是83


#4.进行特征筛选，再重新进行预测 这里选出了32个小于100的特征，删除
feat_imp = pd.Series(clf.booster().get_fscore()).sort_values(ascending=False)
plot_importance(clf)
feature.drop(['xftonAndSumType','disEightVar','disThreeVar','disSevenVar',
'lastNineDis','disSevenMin','disNine','disSeven','disThree','nineAndSumType',
'disThreeMin','disEightMin','disSevenMean','disEight','disSixVar','disEightMax',
'lastEightDis','varTime','disTwoVar','disThreeMax','disTwoMax','disSevenMax',
'disTwoMin','disFourMax','disEightMean','disSixMean','disSixMin','disFourMin',
'ThreeAndSumType','eightAndNine','disNineMean','disNineVar'], axis=1, inplace=True)

#重复上面步骤，进行训练预测，
#去掉32个干扰特征后，准确率有所上升，现在是84%
#采用朴素随机过采样，全部特征的准确率高于特征选择准确率。现在是87%应该继续从提特征入手。

#保存feature特征
feature.to_csv(r'E:\data\黄包车比赛\feature46.csv', index=False)



#使用catboost进行训练预测
from catboost import CatBoostClassifier
model = CatBoostClassifier(loss_function='Logloss',eval_metric='AUC')
model = CatBoostClassifier(iterations=500, depth=5, learning_rate=0.06, loss_function='Logloss',eval_metric='AUC')
model.fit(X_resampled, y_resampled, eval_set=(X_test, y_test), plot=True)
model.fit(X_train, y_train, eval_set=(X_test, y_test), plot=True)


#对验证集进行预测
preds_proba = model.predict_proba(X_test)

accuracy = roc_auc_score(y_test, preds_proba[:,1:])
print("Accuracy: %.2f%%" % (accuracy * 100.0)) #  95.77%         95.83%     101:95.59%

#对测试集进行预测
test_preds_proba = model.predict_proba(test)
a = pd.DataFrame(test_preds_proba, columns=['as','asd'])

data = {'userid':userid, 'orderType':a['asd'].values}
submit = pd.DataFrame(data, columns=['userid', 'orderType'])
submit.to_csv(r'E:\data\黄包车比赛\皇包车比赛\submit45.csv', index=False)  #准确率是83

#绘制学习曲线
def plot_learning_curve(estimator, title, X, y,ylim=None,          cv=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):

    '''
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    '''
    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=1, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid("on") 
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.show()
     

plot_learning_curve(clf, "xbgoost",
                    X_resampled.values, y_resampled['orderType'].values, ylim=(0.7, 1.01),
                    train_sizes=np.linspace(.05, 0.2, 5))








#换成随机森林试一试

from sklearn.ensemble import RandomForestClassifier

rf =RandomForestClassifier(n_estimators=200, max_depth=5, oob_score=True)
rf.fit(X_resampled, y_resampled)

print (rf.oob_score_)
predictions_rf = rf.predict(X_test)
accuracy = roc_auc_score(y_test, predictions_rf)
print("Accuracy: %.2f%%" % (accuracy * 100.0))    #90.85%

test_predictions_rf = rf.predict(test)
Counter(test_predictions_rf)

data = {'userid':userid, 'orderType':test_predictions_rf}
submit = pd.DataFrame(data, columns=['userid', 'orderType'])
Counter(submit.orderType)
submit.to_csv(r'E:\data\黄包车比赛\皇包车比赛\submit7.csv', index=False)

#结果比xgboost差一点


#换成svm试一试

#数据归一化
min_max = MinMaxScaler()
X_train = min_max.fit_transform(X_train)

X_test = min_max.fit_transform(X_test)


from sklearn.svm import SVC
svc = SVC()
svc.fit(X_resampled, y_resampled)

predictions_svc = svc.predict(X_test)
accuracy = roc_auc_score(y_test, predictions_svc)
print("Accuracy: %.2f%%" % (accuracy * 100.0))  

test_predictions_svc = svc.predict(test)
Counter(test_predictions_svc)

data = {'userid':userid, 'orderType':test_predictions_svc}
submit = pd.DataFrame(data, columns=['userid', 'orderType'])
Counter(submit.orderType)
submit.to_csv(r'E:\data\黄包车比赛\皇包车比赛\submit9.csv', index=False)

#结果十分差！！！！！！！mmmmmmp


#接下来进行调参
#
#


params = {
    'learning_rate':0.05,
    'n_estimators':500,
    'max_depth':5,
    'min_child_weight':1,
    'gamma':0,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'objective':'binary:logistic',
    #在各类别样本十分不平衡时，把这个参数设定为正值，可以是算法更快收敛
    'scale_pos_weight':1
}

clf3 = xgb.XGBClassifier(**params)

#首先调整学习率
grid_params = {
    'learning_rate':[0.06, 0.07, 0.08]  
}

grid = GridSearchCV(clf3,grid_params)
grid.fit(X_resampled,y_resampled)

print(grid.best_params_)
print("Accuracy:{0:.1f}%".format(100*grid.best_score_))
#{'learning_rate': 0.06}
#Accuracy:89.4%

#调整树的数目
params = {
    'learning_rate':0.06,
    'n_estimators':500,
    'max_depth':5,
    'min_child_weight':1,
    'gamma':0,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'objective':'binary:logistic',
    #在各类别样本十分不平衡时，把这个参数设定为正值，可以是算法更快收敛
    'scale_pos_weight':1
}

clf4 = xgb.XGBClassifier(**params)
grid_params = {
    # 'learning_rate':np.linspace(0.01,0.2,20),  #得到最佳参数0.01，Accuracy：96.4%
     'n_estimators':list(range(200,501,100)),  #得到最佳参数500，Accuracy：96.4%
}

grid = GridSearchCV(clf4,grid_params)
grid.fit(X_resampled,y_resampled)
#{'n_estimators': 200}
#Accuracy:89.6%





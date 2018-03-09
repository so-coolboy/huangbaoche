# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 10:52:38 2017

@author: www
"""
#本文件用于对数据进行预处理



import pandas as pd
import numpy as np
from collections import Counter

#读入action表的训练集和测试集
action_train = pd.read_csv(r'E:\data\黄包车比赛\皇包车比赛\皇包车比赛数据-非压缩包\trainingset\action_train.csv')
action_test = pd.read_csv(r'E:\data\黄包车比赛\皇包车比赛\皇包车比赛数据-非压缩包\test\action_test.csv')

#读入orderHistory_train表数据
orderHistory_train = pd.read_csv(r'E:\data\黄包车比赛\皇包车比赛\皇包车比赛数据-非压缩包\trainingset\orderHistory_train.csv')

#把训练集和测试集组合起来，便于同时提取特征
df = pd.concat((action_train, action_test), axis=0)


import time
def time_conv(x):
    timeArray=time.localtime(x)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime
    
df.actionTime = pd.to_datetime(df.actionTime.map(lambda x: time_conv(x)),format="%Y-%m-%d %H:%M:%S")
orderHistory_train.orderTime = pd.to_datetime(orderHistory_train.orderTime.map(lambda x: time_conv(x)),format="%Y-%m-%d %H:%M:%S")

#把年月日时组合起来作为一个特征
def fun_ymdh(arr):
     years = str(arr.year)
     months = str(arr.month)
     days = str(arr.day)
     hours = str(arr.hour)
     if len(months) == 1:
          months = '0' + months
     if len(days) == 1:
          days = '0' + days
     if len(hours) == 1:
          hours = '0' + hours
     a = int (years + months + days + hours)
     return a

df ['ymdh'] = df['actionTime'].apply(fun_ymdh)


#把分转化为秒。记录总的秒数
def fun_ms(arr):
     a = arr.minute * 60 + arr.second
     return a

df['minSec'] = df['actionTime'].apply(fun_ms)     


#保存df
df.to_csv(r'E:\data\黄包车比赛\df.csv', index=False)
df = pd.read_csv(r'E:\data\黄包车比赛\df.csv')


#提取特征
feature = pd.DataFrame()

group1 = df.groupby('userid')
group2 = df.groupby(['userid','ymdh'])

#第一个特征 ，按天数每个用户操作的次数
def fun1(arr):
     group2 = arr.groupby('ymdh')
     return len(group2.size())
     
feature['count1'] = group1.apply(fun1)


#第二个特征，用户操作中大于等于5的次数
def fun2(arr):
     group2 = arr.groupby('ymdh').size()
     count = 0
     for i in group2:
          if i >= 5:
               count += 1
     return count

feature['cGreatF'] = group1.apply(fun2)    #13357


#第三个特征，用户操作中大于等于7的次数
def fun3(arr):
     group2 = arr.groupby('ymdh').size()
     count = 0
     for i in group2:
          if i >= 7:
               count += 1
     return count

feature['cGreatSeven'] = group1.apply(fun3)   #a = 24535


#第四个特征，用户操作中大于等于9的次数
def fun4(arr):
     group2 = arr.groupby('ymdh').size()
     count = 0
     for i in group2:
          if i >= 9:
               count += 1
     return count

feature['cGreatNine'] = group1.apply(fun4)
a = feature[feature['cGreatNine']==0].sum(axis=1) #a = 33409  

#第五个特征，用户操作中大于等于11的次数
def fun5(arr):
     group2 = arr.groupby('ymdh').size()
     count = 0
     for i in group2:
          if i >= 11:
               count += 1
     return count

feature['cGreatFF'] = group1.apply(fun5)

a = feature[feature['cGreatFF']==0].sum(axis=1)  #a = 38662


#第六个特征，用户操作中大于等于13的次数
def fun6(arr):
     group2 = arr.groupby('ymdh').size()
     count = 0
     for i in group2:
          if i >= 13:
               count += 1
     return count

feature['cGreatFTh'] = group1.apply(fun6)

a = feature[feature['cGreatFTh']==0].sum(axis=1)  #a=42295


#第七个特征，用户操作中大于等于5的次数占总操作的比例
def fun7(arr):
     group2 = arr.groupby('ymdh').size()
     count = 0
     for i in group2:
          if i >= 5:
               count += 1
     return count/len(group2)

feature['cGreatFxxx'] = group1.apply(fun7) 


#第8个特征，用户操作中大于等于7的次数占总操作的比例
def fun8(arr):
     group2 = arr.groupby('ymdh').size()
     count = 0
     for i in group2:
          if i >= 7:
               count += 1
     return count/len(group2)

feature['cGreatSxxx'] = group1.apply(fun8) 


#第九个特征，用户操作中大于等于9的次数占总操作的比例
def fun9(arr):
     group2 = arr.groupby('ymdh').size()
     count = 0
     for i in group2:
          if i >= 9:
               count += 1
     return count/len(group2)

feature['cGreatNxxx'] = group1.apply(fun9) 


#第十个特征，用户操作中大于等于11的次数占总操作的比例
def fun10(arr):
     group2 = arr.groupby('ymdh').size()
     count = 0
     for i in group2:
          if i >= 11:
               count += 1
     return count/len(group2)

feature['cGreatNnxx'] = group1.apply(fun10)


#第十一个特征，用户操作中大于等于13的次数占总操作的比例
def fun11(arr):
     group2 = arr.groupby('ymdh').size()
     count = 0
     for i in group2:
          if i >= 13:
               count += 1
     return count/len(group2)

feature['cGreatNThxx'] = group1.apply(fun11)


#第十二个特征，用户大于5的操作中，每次操作的（均值，方差，最小值）的平均，最小值
#第十二个特征，用户大于5的操作中，每次操作的时间最小值
def fun12(arr):
     group2 = arr.groupby('ymdh')
     temp = []
     for name, group in group2:
          if len(group) >= 5:
               d = group['minSec'].max() - group['minSec'].min()
               temp.append(d)
     if len(temp)==0:
          return 0
     else:
          temp = np.array(temp)
          return np.min(temp)
     
feature['interFFF'] = group1.apply(fun12)
     

#第十三个特征，用户大于5的操作操作中，每次操作的时间最大值
def fun13(arr):
     group2 = arr.groupby('ymdh')
     temp = []
     for name, group in group2:
          if len(group) >= 5:
               d = group['minSec'].max() - group['minSec'].min()
               temp.append(d)
     if len(temp)==0:
          return 0
     else:
          temp = np.array(temp)
          return np.max(temp)
     
feature['interMMM'] = group1.apply(fun13)



#第十四个特征，用户大于5的操作操作中，每次操作的时间平均值
def fun14(arr):
     group2 = arr.groupby('ymdh')
     temp = []
     for name, group in group2:
          if len(group) >= 5:
               d = group['minSec'].max() - group['minSec'].min()
               temp.append(d)
     if len(temp)==0:
          return 0
     else:
          temp = np.array(temp)
          return np.mean(temp)
     
feature['interMMean'] = group1.apply(fun14)


#第十五个特征，用户大于5的操作操作中，最多的操作次数
def fun15(arr):
     group2 = arr.groupby('ymdh')
     temp = []
     for name, group in group2:
          if len(group) >= 5:
               d = len(group)
               temp.append(d)
     if len(temp)==0:
          return 0
     else:
          temp = np.array(temp)
          return np.max(temp)
     
feature['interMTTime'] = group1.apply(fun15)


#第十六个特征，用户大于5的操作操作中，平均的操作次数
def fun16(arr):
     group2 = arr.groupby('ymdh')
     temp = []
     for name, group in group2:
          if len(group) >= 5:
               d = len(group)
               temp.append(d)
     if len(temp)==0:
          return 0
     else:
          temp = np.array(temp)
          return np.mean(temp)
     
feature['interMTTeann'] = group1.apply(fun16)



#第十七个特征，用户大于5的操作中，每次操作的时间除以次数
def fun17(arr):
     group2 = arr.groupby('ymdh')
     temp = []
     for name, group in group2:
          if len(group) >= 5:
               d = group['minSec'].max() - group['minSec'].min()
               temp.append(d/len(group))
     if len(temp)==0:
          return 0
     else:
          temp = np.array(temp)
          return np.max(temp)
     
feature['temp_17'] = group1.apply(fun17)


#第十八个特征，用户大于5的操作中，每轮操作中，间隔最小的的最小值
def fun18(arr):
     group2 = arr.groupby('ymdh')
     temp1 = []
     for name, group in group2:
          if len(group) >= 5:
               temp2 = []
               for i in range(len(group)-1):
                    m = group[i+1:i+2]['minSec'].values - group[i:i+1]['minSec'].values
                    temp2.append(m)
               temp2 = np.array(temp2)
               n = np.min(temp2)
          
               temp1.append(n)
     if len(temp1) == 0:
          return 0
     else:
          temp1 = np.array(temp1)
          return np.min(temp1)

feature['temp_18'] = group1.apply(fun18)


#第十九个特征用户大于5等于5的操作中，梅伦操作中，间隔最da值的最大值，
def fun19(arr):
     group2 = arr.groupby('ymdh')
     temp1 = []
     for name, group in group2:
          if len(group) >= 5:
               temp2 = []
               for i in range(len(group)-1):
                    m = group[i+1:i+2]['minSec'].values - group[i:i+1]['minSec'].values
                    temp2.append(m)
               temp2 = np.array(temp2)
               n = np.max(temp2)
               temp1.append(n)
     if len(temp1) == 0:
          return 0
     else:
          temp1 = np.array(temp1)
          return np.max(temp1)

feature['temp_19'] = group1.apply(fun19)    


#第二十个特征用户大于5等于5的操作中，梅伦操作中，间隔最小值的最大值，
def fun20(arr):
     group2 = arr.groupby('ymdh')
     temp1 = []
     for name, group in group2:
          if len(group) >= 5:
               temp2 = []
               for i in range(len(group)-1):
                    m = group[i+1:i+2]['minSec'].values - group[i:i+1]['minSec'].values
                    temp2.append(m)
               temp2 = np.array(temp2)
               n = np.min(temp2)
               temp1.append(n)
     if len(temp1) == 0:
          return 0
     else:
          temp1 = np.array(temp1)
          return np.max(temp1)

feature['temp_20'] = group1.apply(fun20)   


feature.to_csv(r'E:\data\黄包车比赛\feature20.csv', index=False)
feature = pd.read_csv(r'E:\data\黄包车比赛\feature16.csv')


df.to_csv(r'E:\data\黄包车比赛\df.csv', index=False)
df = pd.read_csv(r'E:\data\黄包车比赛\df.csv')













#
#...................对action表的处理：
#
grouped = df.groupby('userid')

grouped_id = df.groupby('userid')

#第一个特征： 每个用户的行为信息数量 count
feature['count'] = grouped.size()


#第二个特征，操作的总时间长
def fun2(arr):
    return arr.max() - arr.min()

feature['sumTime'] = grouped['actionTime'].agg(fun2)


#第三个特征，每次操作的平均时间，即时间间隔的均值
feature['meanTime'] = feature['sumTime']/(feature['count'].values-1)
feature['meanTime'].fillna(0, inplace=True)


#第四个特征，标准差，即时间间隔的标准差
def fun4(arr):
    if len(arr)==1:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = np.array(temp)
        return np.std(temp1)
    else:
        return 0
    
feature['stdtime'] = grouped['actionTime'].apply(fun4)


#第五个特征，方差，即时间间隔的方差
def fun5(arr):
    if len(arr)==1:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = np.array(temp)
        return np.var(temp1)
    else:
        return 0
    
feature['varTime'] = grouped['actionTime'].apply(fun5)


#第六个特征，时间间隔的最小值
def fun6(arr):
    if len(arr)==1:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = np.array(temp)
        return temp1[temp1.argmin()]
    else:
        return 0
    
feature['minIntTime'] = grouped['actionTime'].apply(fun6)


#第七个特征，时间间隔的末尾值
def fun7(arr):
    if len(arr)==1:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = np.array(temp)
        return temp1[-1]
    else:
        return 0
    
feature['lastIntTime'] = grouped['actionTime'].apply(fun7)


#第八个特征，时间间隔倒数第二个值
def fun8(arr):
    if len(arr)<3:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = np.array(temp)
        return temp1[-2]
    else:
        return 0
    
feature['lastTwoIntTime'] = grouped['actionTime'].apply(fun8)


#第九个特征，时间间隔倒数第三个值
def fun9(arr):
    if len(arr)<4:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = np.array(temp)
        return temp1[-3]
    else:
        return 0
    
feature['lastThreeIntTime'] = grouped['actionTime'].apply(fun9)


#第十个特征，时间间隔倒数第四个值
def fun10(arr):
    if len(arr)<5:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = np.array(temp)
        return temp1[-4]
    else:
        return 0
    
feature['lastFourIntTime'] = grouped['actionTime'].apply(fun10)


#第十一个特征，倒数最后一个type
def fun11(arr):
    if len(arr)==1:
        return int(arr)
    else:
        arr1 = np.array(arr)
        return arr1[-1]
    
feature['lastType'] = grouped['actionType'].apply(fun11)


#第十二个特征，倒数第二个type
def fun12(arr):
    if len(arr)==1:
        return 0
    else:
        arr1 = np.array(arr)
        return arr1[-2]
    
feature['lastTwoType'] = grouped['actionType'].apply(fun12)


#第十三个特征，倒数第三个type
def fun13(arr):
    if len(arr)<=2:
        return 0
    else:
        arr1 = np.array(arr)
        return arr1[-3]
    
feature['lastThreeType'] = grouped['actionType'].apply(fun13)


#第十四个特征，最后三个时间间隔均值
def fun14(arr):
    if len(arr)<4:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = np.array(temp)
        return (temp1[-1] + temp1[-2] + temp1[-3])/3
    else:
        return 0

feature['lastThreeInterMean'] = grouped['actionTime'].apply(fun14)


#第十五个特征，最后三个时间间隔的方差
def fun15(arr):
    if len(arr)<4:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = np.array(temp)
        temp2 = np.array([temp1[-1] , temp1[-2] , temp1[-3]])
        return np.var(temp2)
    else:
        return 0
    
feature['lastThreeInterVar'] = grouped['actionTime'].apply(fun15)


#第十六个特征，第一个type
def fun16(arr):
    if len(arr)==1:
        return int(arr)
    else:
        arr1 = np.array(arr)
        return arr1[0]
    
feature['firstType'] = grouped['actionType'].apply(fun16)


#第十七个特征，第一个时间间隔
def fun17(arr):
    if len(arr)==1:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = np.array(temp)
        return temp1[0]
    else:
        return 0
    
feature['firstInter'] = grouped['actionTime'].apply(fun17)


#第十八个特征，用户点击1与总点击数之比
def fun18(arr):
    arr1 = np.array(arr)
    count = 0
    for i in arr1:
        if i == 1:
            count+=1
    return count/len(arr1)

feature['firstAndSumType'] = grouped['actionType'].apply(fun18)


#第十九个特征，用户点击2与总点击数相比
def fun19(arr):
    arr1 = np.array(arr)
    count = 0
    for i in arr1:
        if i == 2:
            count+=1
    return count/len(arr1)

feature['TwoAndSumType'] = grouped['actionType'].apply(fun19)


#第二十个特征，用户点击3与总点击数相比
def fun20(arr):
    arr1 = np.array(arr)
    count = 0
    for i in arr1:
        if i == 3:
            count+=1
    return count/len(arr1)

feature['ThreeAndSumType'] = grouped['actionType'].apply(fun20)


#第二十一个特征，用户点击4与总点击数相比
def fun21(arr):
    arr1 = np.array(arr)
    count = 0
    for i in arr1:
        if i == 4:
            count+=1
    return count/len(arr1)

feature['FourAndSumType'] = grouped['actionType'].apply(fun21)


#第二十二个特征，用户点击5与总点击数相比
def fun22(arr):
    arr1 = np.array(arr)
    count = 0
    for i in arr1:
        if i == 5:
            count+=1
    return count/len(arr1)

feature['FiveeAndSumType'] = grouped['actionType'].apply(fun22)


#第二十三个特征，用户点击6与总点击数相比
def fun23(arr):
    arr1 = np.array(arr)
    count = 0
    for i in arr1:
        if i == 6:
            count+=1
    return count/len(arr1)

feature['sixAndSumType'] = grouped['actionType'].apply(fun23)


#第二十四个特征，用户点击7与总点击数相比
def fun24(arr):
    arr1 = np.array(arr)
    count = 0
    for i in arr1:
        if i == 7:
            count+=1
    return count/len(arr1)

feature['sevenAndSumType'] = grouped['actionType'].apply(fun24)


#第二十五个特征，用户点击8与总点击数相比
def fun25(arr):
    arr1 = np.array(arr)
    count = 0
    for i in arr1:
        if i == 8:
            count+=1
    return count/len(arr1)

feature['eightAndSumType'] = grouped['actionType'].apply(fun25)


#第二十六个特征，用户点击9与总点击数相比`
def fun26(arr):
    arr1 = np.array(arr)
    count = 0
    for i in arr1:
        if i == 9:
            count+=1
    return count/len(arr1)

feature['nineAndSumType'] = grouped['actionType'].apply(fun26)


#第二十七个特征，最后的类型在5到9之间的比例占总操作的比例
def fun27(arr):
    arr1 = np.array(arr)
    count = 0
    for i in arr1:
        if i > 4:
            count+=1
    return count/len(arr1)

feature['ftonAndSumType'] = grouped['actionType'].apply(fun27)


#第二十八个特征，到最后一个5的距离
def fun28(arr):
     arr1 = np.array(arr)
     count = 1
     if 5 not in arr1:
          return 0
     for i in arr1[::-1]:
          if i != 5:
               count+=1
          else:
               return count
     
     
feature['lastFiveDis'] = grouped['actionType'].apply(fun28)
 
    
#第二十九个特征，到最后一个6的距离
def fun29(arr):
     arr1 = np.array(arr)
     count = 1
     if 6 not in arr1:
          return 0
     for i in arr1[::-1]:
          if i != 6:
               count+=1
          else:
               return count
     
feature['lastSixDis'] = grouped['actionType'].apply(fun29)     


#第三十个特征，到最后一个8的距离
def fun30(arr):
     arr1 = np.array(arr)
     count = 1
     if 8 not in arr1:
          return 0
     for i in arr1[::-1]:
          if i != 8:
               count+=1
          else:
               return count
     
feature['lastEightDis'] = grouped['actionType'].apply(fun30)


#第三十一个特征，到最后一个9的距离
def fun31(arr):
     arr1 = np.array(arr)
     count = 1
     if 9 not in arr1:
          return 0
     for i in arr1[::-1]:
          if i != 9:
               count+=1
          else:
               return count
     
     
feature['lastNineDis'] = grouped['actionType'].apply(fun31)


#第三十二个特征，是否含有5到9的操作
def fun32(arr):
    arr1 = np.array(arr)
    
    for i in arr1:
        if i > 4:
            return 1
    return 0

feature['xftonAndSumType'] = grouped['actionType'].apply(fun32)


#第三十三个特征，5,6连用的的类型占左右类型的比例
def fun33(arr):
     arr1 = np.array(arr)
     if len(arr) == 1:
          return 0
     count = 0
     for i in range(len(arr1)-1):
          if arr1[i]==5 and arr1[i+1]==6:
               count+=1
     return count/(len(arr1)-1)

feature['fiveAndSix'] = grouped['actionType'].apply(fun33)


#第三十四个特征，8,9连用的的类型占左右类型的比例
def fun34(arr):
     arr1 = np.array(arr)
     if len(arr) == 1:
          return 0
     count = 0
     for i in range(len(arr1)-1):
          if arr1[i]==8 and arr1[i+1]==9:
               count+=1
     return count/(len(arr1)-1)

feature['eightAndNine'] = grouped['actionType'].apply(fun34)    


#第三十五个特征，离最近的1的时间
def fun35(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    m = 0
    a = temp1.head(1)
    for i in range(len(temp1)):
         if temp1[i:i+1]['actionType'].values == 1:
              m = a['actionTime'].values - temp1[i:i+1]['actionTime'].values
              return int(m)
    return m  
              
feature['disFirst'] = grouped.apply(fun35)


#第三十六个特征，离最近的2的时间
def fun36(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    m = 0
    a = temp1.head(1)
    for i in range(len(temp1)):
         if temp1[i:i+1]['actionType'].values == 2:
              m = a['actionTime'].values - temp1[i:i+1]['actionTime'].values
              return int(m)
    return m  
              
feature['disTwo'] = grouped.apply(fun36)


#第三十七个特征，离最近的3的时间
def fun37(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    m = 0
    a = temp1.head(1)
    for i in range(len(temp1)):
         if temp1[i:i+1]['actionType'].values == 3:
              m = a['actionTime'].values - temp1[i:i+1]['actionTime'].values
              return int(m)
    return m  
              
feature['disThree'] = grouped.apply(fun37)


#第三十八个特征，离最近的4的时间
def fun38(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    m = 0
    a = temp1.head(1)
    for i in range(len(temp1)):
         if temp1[i:i+1]['actionType'].values == 4:
              m = a['actionTime'].values - temp1[i:i+1]['actionTime'].values
              return int(m)
    return m  
              
feature['disFour'] = grouped.apply(fun38)


#第三十九个特征，离最近的5的时间
def fun39(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    m = 0
    a = temp1.head(1)
    for i in range(len(temp1)):
         if temp1[i:i+1]['actionType'].values == 5:
              m = a['actionTime'].values - temp1[i:i+1]['actionTime'].values
              return int(m)
    return m  
              
feature['disFive'] = grouped.apply(fun39)


#第四十个特征，离最近的6的时间
def fun40(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    m = 0
    a = temp1.head(1)
    for i in range(len(temp1)):
         if temp1[i:i+1]['actionType'].values == 6:
              m = a['actionTime'].values - temp1[i:i+1]['actionTime'].values
              return int(m)
    return m  
              
feature['disSix'] = grouped.apply(fun40)


#第四十一个特征，离最近的7的时间
def fun41(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    m = 0
    a = temp1.head(1)
    for i in range(len(temp1)):
         if temp1[i:i+1]['actionType'].values == 7:
              m = a['actionTime'].values - temp1[i:i+1]['actionTime'].values
              return int(m)
    return m  
              
feature['disSeven'] = grouped.apply(fun41)


#第四十二个特征，离最近的8的时间
def fun42(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    m = 0
    a = temp1.head(1)
    for i in range(len(temp1)):
         if temp1[i:i+1]['actionType'].values == 8:
              m = a['actionTime'].values - temp1[i:i+1]['actionTime'].values
              return int(m)
    return m  
              
feature['disEight'] = grouped.apply(fun42)


#第四十三个特征，离最近的9的时间
def fun43(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    m = 0
    a = temp1.head(1)
    for i in range(len(temp1)):
         if temp1[i:i+1]['actionType'].values == 9:
              m = a['actionTime'].values - temp1[i:i+1]['actionTime'].values
              return int(m)
    return m  
              
feature['disNine'] = grouped.apply(fun43)


#第四十四个特征，离最近的5的时间间隔均值


def fun44(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    
    m = 0
    temp_list = []
    if 5 not in temp1['actionType'].values:
         return 0
    else:
         for i in range(len(temp1)-1):
              if temp1[i:i+1]['actionType'].values != 5:
                   m = temp1[i+1:i+2]['actionTime'].values - temp1[i:i+1]['actionTime'].values
                   temp_list.append(m)
              else:
                   #print(temp1[i:i+1]['actionTime'].values)
                   break
         if len(temp_list)>0:
              temp_list = np.array(temp_list)
              #print(abs(np.mean(temp_list)))
              return abs(np.mean(temp_list))
         else:
              return 0
         
feature['disFiveVar'] = grouped.apply(fun44)


#第四十五个特征，离最近的6的时间间隔均值

def fun45(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    
    m = 0
    temp_list = []
    if 6 not in temp1['actionType'].values:
         return 0
    else:
         for i in range(len(temp1)-1):
              if temp1[i:i+1]['actionType'].values != 6:
                   m = temp1[i+1:i+2]['actionTime'].values - temp1[i:i+1]['actionTime'].values
                   temp_list.append(m)
              else:
                   #print(temp1[i:i+1]['actionTime'].values)
                   break
         if len(temp_list)>0:
              temp_list = np.array(temp_list)
              #print(abs(np.mean(temp_list)))
              return abs(np.mean(temp_list))
         else:
              return 0
         
feature['disSixVar'] = grouped.apply(fun45)


#第四十六个特征，离最近的7的时间间隔均值

def fun46(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    
    m = 0
    temp_list = []
    if 7 not in temp1['actionType'].values:
         return 0
    else:
         for i in range(len(temp1)-1):
              if temp1[i:i+1]['actionType'].values != 7:
                   m = temp1[i+1:i+2]['actionTime'].values - temp1[i:i+1]['actionTime'].values
                   temp_list.append(m)
              else:
                   #print(temp1[i:i+1]['actionTime'].values)
                   break
         if len(temp_list)>0:
              temp_list = np.array(temp_list)
              #print(abs(np.mean(temp_list)))
              return abs(np.mean(temp_list))
         else:
              return 0
         
feature['disSevenVar'] = grouped.apply(fun46)


#第四十七个特征，离最近的8的时间间隔均值

def fun47(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    
    m = 0
    temp_list = []
    if 8 not in temp1['actionType'].values:
         return 0
    else:
         for i in range(len(temp1)-1):
              if temp1[i:i+1]['actionType'].values != 8:
                   m = temp1[i+1:i+2]['actionTime'].values - temp1[i:i+1]['actionTime'].values
                   temp_list.append(m)
              else:
                   #print(temp1[i:i+1]['actionTime'].values)
                   break
         if len(temp_list)>0:
              temp_list = np.array(temp_list)
              #print(abs(np.mean(temp_list)))
              return abs(np.mean(temp_list))
         else:
              return 0
         
feature['disEightVar'] = grouped.apply(fun47)


#第四十八个特征，离最近的9的时间间隔均值

def fun48(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    
    m = 0
    temp_list = []
    if 9 not in temp1['actionType'].values:
         return 0
    else:
         for i in range(len(temp1)-1):
              if temp1[i:i+1]['actionType'].values != 9:
                   m = temp1[i+1:i+2]['actionTime'].values - temp1[i:i+1]['actionTime'].values
                   temp_list.append(m)
              else:
                   #print(temp1[i:i+1]['actionTime'].values)
                   break
         if len(temp_list)>0:
              temp_list = np.array(temp_list)
              #print(abs(np.mean(temp_list)))
              return abs(np.mean(temp_list))
         else:
              return 0
         
feature['disNineVar'] = grouped.apply(fun48)


#第四十九个特征，离最近的5的时间间隔方差

def fun49(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    
    m = 0
    temp_list = []
    if 5 not in temp1['actionType'].values:
         return 0
    else:
         for i in range(len(temp1)-1):
              if temp1[i:i+1]['actionType'].values != 5:
                   m = temp1[i+1:i+2]['actionTime'].values - temp1[i:i+1]['actionTime'].values
                   temp_list.append(m)
              else:
                   #print(temp1[i:i+1]['actionTime'].values)
                   break
         if len(temp_list)>0:
              temp_list = np.array(temp_list)
              #print(abs(np.mean(temp_list)))
              return abs(np.var(temp_list))
         else:
              return 0
         
feature['disFiveMean'] = grouped.apply(fun49)


#第五十个特征，离最近的6的时间间隔方差

def fun50(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    
    m = 0
    temp_list = []
    if 6 not in temp1['actionType'].values:
         return 0
    else:
         for i in range(len(temp1)-1):
              if temp1[i:i+1]['actionType'].values != 6:
                   m = temp1[i+1:i+2]['actionTime'].values - temp1[i:i+1]['actionTime'].values
                   temp_list.append(m)
              else:
                   #print(temp1[i:i+1]['actionTime'].values)
                   break
         if len(temp_list)>0:
              temp_list = np.array(temp_list)
              #print(abs(np.mean(temp_list)))
              return abs(np.var(temp_list))
         else:
              return 0
         
feature['disSixMean'] = grouped.apply(fun50)


#第五十一个特征，离最近的7的时间间隔方差

def fun51(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    
    m = 0
    temp_list = []
    if 7 not in temp1['actionType'].values:
         return 0
    else:
         for i in range(len(temp1)-1):
              if temp1[i:i+1]['actionType'].values != 7:
                   m = temp1[i+1:i+2]['actionTime'].values - temp1[i:i+1]['actionTime'].values
                   temp_list.append(m)
              else:
                   #print(temp1[i:i+1]['actionTime'].values)
                   break
         if len(temp_list)>0:
              temp_list = np.array(temp_list)
              #print(abs(np.mean(temp_list)))
              return abs(np.var(temp_list))
         else:
              return 0
         
feature['disSevenMean'] = grouped.apply(fun51)


#第五十二个特征，离最近的8的时间间隔方差

def fun52(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    
    m = 0
    temp_list = []
    if 8 not in temp1['actionType'].values:
         return 0
    else:
         for i in range(len(temp1)-1):
              if temp1[i:i+1]['actionType'].values != 8:
                   m = temp1[i+1:i+2]['actionTime'].values - temp1[i:i+1]['actionTime'].values
                   temp_list.append(m)
              else:
                   #print(temp1[i:i+1]['actionTime'].values)
                   break
         if len(temp_list)>0:
              temp_list = np.array(temp_list)
              #print(abs(np.mean(temp_list)))
              return abs(np.var(temp_list))
         else:
              return 0
         
feature['disEightMean'] = grouped.apply(fun52)


#第五十三个特征，离最近的9的时间间隔方差

def fun53(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    
    m = 0
    temp_list = []
    if 9 not in temp1['actionType'].values:
         return 0
    else:
         for i in range(len(temp1)-1):
              if temp1[i:i+1]['actionType'].values != 9:
                   m = temp1[i+1:i+2]['actionTime'].values - temp1[i:i+1]['actionTime'].values
                   temp_list.append(m)
              else:
                   #print(temp1[i:i+1]['actionTime'].values)
                   break
         if len(temp_list)>0:
              temp_list = np.array(temp_list)
              #print(abs(np.mean(temp_list)))
              return abs(np.var(temp_list))
         else:
              return 0
         
feature['disNineMean'] = grouped.apply(fun53)


#第五十四个特征，离最近的2的时间间隔方差

def fun54(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    
    m = 0
    temp_list = []
    if 2 not in temp1['actionType'].values:
         return 0
    else:
         for i in range(len(temp1)-1):
              if temp1[i:i+1]['actionType'].values != 2:
                   m = temp1[i+1:i+2]['actionTime'].values - temp1[i:i+1]['actionTime'].values
                   temp_list.append(m)
              else:
                   #print(temp1[i:i+1]['actionTime'].values)
                   break
         if len(temp_list)>0:
              temp_list = np.array(temp_list)
              #print(abs(np.mean(temp_list)))
              return abs(np.var(temp_list))
         else:
              return 0
         
feature['disTwoVar'] = grouped.apply(fun54)


#第五十五个特征，离最近的3的时间间隔方差

def fun55(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    
    m = 0
    temp_list = []
    if 3 not in temp1['actionType'].values:
         return 0
    else:
         for i in range(len(temp1)-1):
              if temp1[i:i+1]['actionType'].values != 3:
                   m = temp1[i+1:i+2]['actionTime'].values - temp1[i:i+1]['actionTime'].values
                   temp_list.append(m)
              else:
                   #print(temp1[i:i+1]['actionTime'].values)
                   break
         if len(temp_list)>0:
              temp_list = np.array(temp_list)
              #print(abs(np.mean(temp_list)))
              return abs(np.var(temp_list))
         else:
              return 0
         
feature['disThreeVar'] = grouped.apply(fun55)


#第五十六个特征，离最近的2的时间间隔最大

def fun56(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    
    m = 0
    temp_list = []
    if 2 not in temp1['actionType'].values:
         return 0
    else:
         for i in range(len(temp1)-1):
              if temp1[i:i+1]['actionType'].values != 2:
                   m = temp1[i+1:i+2]['actionTime'].values - temp1[i:i+1]['actionTime'].values
                   temp_list.append(m)
              else:
                   #print(temp1[i:i+1]['actionTime'].values)
                   break
         if len(temp_list)>0:
              temp_list = np.array(temp_list)
              #print(abs(np.mean(temp_list)))
              return abs(np.min(temp_list))
         else:
              return 0
         
feature['disTwoMin'] = grouped.apply(fun56)


#第五十七个特征，离最近的3的时间间隔最大

def fun57(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    
    m = 0
    temp_list = []
    if 3 not in temp1['actionType'].values:
         return 0
    else:
         for i in range(len(temp1)-1):
              if temp1[i:i+1]['actionType'].values != 3:
                   m = temp1[i+1:i+2]['actionTime'].values - temp1[i:i+1]['actionTime'].values
                   temp_list.append(m)
              else:
                   #print(temp1[i:i+1]['actionTime'].values)
                   break
         if len(temp_list)>0:
              temp_list = np.array(temp_list)
              #print(abs(np.mean(temp_list)))
              return abs(np.min(temp_list))
         else:
              return 0
         
feature['disThreeMin'] = grouped.apply(fun57)


#第五十八个特征，离最近的4的时间间隔最大

def fun58(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    
    m = 0
    temp_list = []
    if 4 not in temp1['actionType'].values:
         return 0
    else:
         for i in range(len(temp1)-1):
              if temp1[i:i+1]['actionType'].values != 4:
                   m = temp1[i+1:i+2]['actionTime'].values - temp1[i:i+1]['actionTime'].values
                   temp_list.append(m)
              else:
                   #print(temp1[i:i+1]['actionTime'].values)
                   break
         if len(temp_list)>0:
              temp_list = np.array(temp_list)
              #print(abs(np.mean(temp_list)))
              return abs(np.min(temp_list))
         else:
              return 0
         
feature['disFourMin'] = grouped.apply(fun58)


#第五十九个特征，离最近的5的时间间隔最大

def fun59(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    
    m = 0
    temp_list = []
    if 5 not in temp1['actionType'].values:
         return 0
    else:
         for i in range(len(temp1)-1):
              if temp1[i:i+1]['actionType'].values != 5:
                   m = temp1[i+1:i+2]['actionTime'].values - temp1[i:i+1]['actionTime'].values
                   temp_list.append(m)
              else:
                   #print(temp1[i:i+1]['actionTime'].values)
                   break
         if len(temp_list)>0:
              temp_list = np.array(temp_list)
              #print(abs(np.mean(temp_list)))
              return abs(np.min(temp_list))
         else:
              return 0
         
feature['disFiveMin'] = grouped.apply(fun59)


#第六十个特征，离最近的6的时间间隔最大

def fun60(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    
    m = 0
    temp_list = []
    if 6 not in temp1['actionType'].values:
         return 0
    else:
         for i in range(len(temp1)-1):
              if temp1[i:i+1]['actionType'].values != 6:
                   m = temp1[i+1:i+2]['actionTime'].values - temp1[i:i+1]['actionTime'].values
                   temp_list.append(m)
              else:
                   #print(temp1[i:i+1]['actionTime'].values)
                   break
         if len(temp_list)>0:
              temp_list = np.array(temp_list)
              #print(abs(np.mean(temp_list)))
              return abs(np.min(temp_list))
         else:
              return 0
         
feature['disSixMin'] = grouped.apply(fun60)



#第六十一个特征，离最近的7的时间间隔最大

def fun61(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    
    m = 0
    temp_list = []
    if 7 not in temp1['actionType'].values:
         return 0
    else:
         for i in range(len(temp1)-1):
              if temp1[i:i+1]['actionType'].values != 7:
                   m = temp1[i+1:i+2]['actionTime'].values - temp1[i:i+1]['actionTime'].values
                   temp_list.append(m)
              else:
                   #print(temp1[i:i+1]['actionTime'].values)
                   break
         if len(temp_list)>0:
              temp_list = np.array(temp_list)
              #print(abs(np.mean(temp_list)))
              return abs(np.min(temp_list))
         else:
              return 0
         
feature['disSevenMin'] = grouped.apply(fun61)


#第六十二个特征，离最近的8的时间间隔最小大

def fun62(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    
    m = 0
    temp_list = []
    if 8 not in temp1['actionType'].values:
         return 0
    else:
         for i in range(len(temp1)-1):
              if temp1[i:i+1]['actionType'].values != 8:
                   m = temp1[i+1:i+2]['actionTime'].values - temp1[i:i+1]['actionTime'].values
                   temp_list.append(m)
              else:
                   #print(temp1[i:i+1]['actionTime'].values)
                   break
         if len(temp_list)>0:
              temp_list = np.array(temp_list)
              #print(abs(np.mean(temp_list)))
              return abs(np.min(temp_list))
         else:
              return 0
         
feature['disEightMin'] = grouped.apply(fun62)


#第六十三个特征，离最近的2的时间间隔最小

def fun63(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    
    m = 0
    temp_list = []
    if 2 not in temp1['actionType'].values:
         return 0
    else:
         for i in range(len(temp1)-1):
              if temp1[i:i+1]['actionType'].values != 2:
                   m = temp1[i+1:i+2]['actionTime'].values - temp1[i:i+1]['actionTime'].values
                   temp_list.append(m)
              else:
                   #print(temp1[i:i+1]['actionTime'].values)
                   break
         if len(temp_list)>0:
              temp_list = np.array(temp_list)
              #print(abs(np.mean(temp_list)))
              return abs(np.max(temp_list))
         else:
              return 0
         
feature['disTwoMax'] = grouped.apply(fun63)


#第六十四个特征，离最近的3的时间间隔最小

def fun64(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    
    m = 0
    temp_list = []
    if 3 not in temp1['actionType'].values:
         return 0
    else:
         for i in range(len(temp1)-1):
              if temp1[i:i+1]['actionType'].values != 3:
                   m = temp1[i+1:i+2]['actionTime'].values - temp1[i:i+1]['actionTime'].values
                   temp_list.append(m)
              else:
                   #print(temp1[i:i+1]['actionTime'].values)
                   break
         if len(temp_list)>0:
              temp_list = np.array(temp_list)
              #print(abs(np.mean(temp_list)))
              return abs(np.max(temp_list))
         else:
              return 0
         
feature['disThreeMax'] = grouped.apply(fun64)


#第六十五个特征，离最近的4的时间间隔最小

def fun65(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    
    m = 0
    temp_list = []
    if 4 not in temp1['actionType'].values:
         return 0
    else:
         for i in range(len(temp1)-1):
              if temp1[i:i+1]['actionType'].values != 4:
                   m = temp1[i+1:i+2]['actionTime'].values - temp1[i:i+1]['actionTime'].values
                   temp_list.append(m)
              else:
                   #print(temp1[i:i+1]['actionTime'].values)
                   break
         if len(temp_list)>0:
              temp_list = np.array(temp_list)
              #print(abs(np.mean(temp_list)))
              return abs(np.max(temp_list))
         else:
              return 0
         
feature['disFourMax'] = grouped.apply(fun65)


#第六十六个特征，离最近的5的时间间隔最小

def fun66(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    
    m = 0
    temp_list = []
    if 5 not in temp1['actionType'].values:
         return 0
    else:
         for i in range(len(temp1)-1):
              if temp1[i:i+1]['actionType'].values != 5:
                   m = temp1[i+1:i+2]['actionTime'].values - temp1[i:i+1]['actionTime'].values
                   temp_list.append(m)
              else:
                   #print(temp1[i:i+1]['actionTime'].values)
                   break
         if len(temp_list)>0:
              temp_list = np.array(temp_list)
              #print(abs(np.mean(temp_list)))
              return abs(np.max(temp_list))
         else:
              return 0
         
feature['disFiveMax'] = grouped.apply(fun66)


#第六十七个特征，离最近的6的时间间隔最小

def fun67(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    
    m = 0
    temp_list = []
    if 6 not in temp1['actionType'].values:
         return 0
    else:
         for i in range(len(temp1)-1):
              if temp1[i:i+1]['actionType'].values != 6:
                   m = temp1[i+1:i+2]['actionTime'].values - temp1[i:i+1]['actionTime'].values
                   temp_list.append(m)
              else:
                   #print(temp1[i:i+1]['actionTime'].values)
                   break
         if len(temp_list)>0:
              temp_list = np.array(temp_list)
              #print(abs(np.mean(temp_list)))
              return abs(np.max(temp_list))
         else:
              return 0
         
feature['disSixMax'] = grouped.apply(fun67)


#第六十八个特征，离最近的7的时间间隔最小

def fun68(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    
    m = 0
    temp_list = []
    if 7 not in temp1['actionType'].values:
         return 0
    else:
         for i in range(len(temp1)-1):
              if temp1[i:i+1]['actionType'].values != 7:
                   m = temp1[i+1:i+2]['actionTime'].values - temp1[i:i+1]['actionTime'].values
                   temp_list.append(m)
              else:
                   #print(temp1[i:i+1]['actionTime'].values)
                   break
         if len(temp_list)>0:
              temp_list = np.array(temp_list)
              #print(abs(np.mean(temp_list)))
              return abs(np.max(temp_list))
         else:
              return 0
         
feature['disSevenMax'] = grouped.apply(fun68)


#第六十九个特征，离最近的8的时间间隔最小

def fun69(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    
    m = 0
    temp_list = []
    if 8 not in temp1['actionType'].values:
         return 0
    else:
         for i in range(len(temp1)-1):
              if temp1[i:i+1]['actionType'].values != 8:
                   m = temp1[i+1:i+2]['actionTime'].values - temp1[i:i+1]['actionTime'].values
                   temp_list.append(m)
              else:
                   #print(temp1[i:i+1]['actionTime'].values)
                   break
         if len(temp_list)>0:
              temp_list = np.array(temp_list)
              #print(abs(np.mean(temp_list)))
              return abs(np.max(temp_list))
         else:
              return 0
         
feature['disEightMax'] = grouped.apply(fun69)


#第七十个特征，时间间隔的次小值

def fun70(arr):
    if len(arr)<3:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = np.sort(temp)
        return temp1[1]
    else:
        return 0
    
feature['twoMinInterTime'] = grouped['actionTime'].apply(fun70)


#第七十一个特征，时间间隔的第三小值

def fun71(arr):
    if len(arr)<4:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = np.sort(temp)
        return temp1[2]
    else:
        return 0
    
feature['threeMinInterTime'] = grouped['actionTime'].apply(fun71)


#第七十二个特征，时间间隔的第四小值

def fun72(arr):
    if len(arr)<5:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = np.sort(temp)
        return temp1[3]
    else:
        return 0
    
feature['fourMinInterTime'] = grouped['actionTime'].apply(fun72)


#第七十三个特征，时间间隔后四个的均值

def fun73(arr):
    if len(arr)<5:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = temp[-4:]
        return np.mean(temp1)
    else:
        return 0
    
feature['fourMeanInterTime'] = grouped['actionTime'].apply(fun73)


#第七十四个特征，时间间隔后四个的最大值

def fun74(arr):
    if len(arr)<5:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = temp[-4:]
        return np.max(temp1)
    else:
        return 0
    
feature['fourMaxInterTime'] = grouped['actionTime'].apply(fun74)


#第七十五个特征，时间间隔后四个的最小值

def fun75(arr):
    if len(arr)<5:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = temp[-4:]
        return np.min(temp1)
    else:
        return 0
    
feature['fourInterMinTime'] = grouped['actionTime'].apply(fun75)


#第七十六个特征，2到4操作次数占总操作次数之比

def fun76(arr):
    arr1 = np.array(arr)
    count = 0
    for i in arr1:
        if i == 4 or i == 2 or i == 3:
            count+=1
    return count/len(arr1)

feature['twoandFourType'] = grouped['actionType'].apply(fun76)


#第七十七个特征，5,6操作次数占总操作次数之比

def fun77(arr):
    arr1 = np.array(arr)
    count = 0
    for i in arr1:
        if i == 5 or i == 6:
            count+=1
    return count/len(arr1)

feature['fiveandSixType'] = grouped['actionType'].apply(fun77)


#==============================================================================
# #与5,6.7.8有关的时间间隔的均值
# 
# def fun78(df,columns = 'actionTime'):
#     temp_list = []
#     for i in range(len(df)-1):
#         a = df[i:i+1]['actionType'].values
#         if a == 5 or a == 6 or a==7 or a==8:
#             m = df[i+1:i+2]['actionTime'].values - df[i:i+1]['actionTime'].values
#             temp_list.append(m)
#     if len(temp_list)>0:
#         temp_list = np.array(temp_list)
#               #print(abs(np.mean(temp_list)))
#         return abs(np.mean(temp_list))
#     else:
#         return 0
#      
# feature['fivetoeightMean'] = grouped.apply(fun78)
#==============================================================================

feature['userid'] = feature.index


feature.to_csv(r'E:\data\黄包车比赛\feature77.csv',index = False)


#第七十八个值，时间间隔后三个的均值
def fun78(arr):
    if len(arr)<4:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = temp[-3:]
        return np.mean(temp1)
    else:
        return 0
    
feature['threexMeanInterTime'] = grouped['actionTime'].apply(fun78)


#第七十九个值，时间间隔后三个的最小值
def fun79(arr):
    if len(arr)<4:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = temp[-3:]
        return np.min(temp1)
    else:
        return 0
    
feature['threexMinInterTime'] = grouped['actionTime'].apply(fun79)


#第八十个值，时间间隔后三个的最大值
def fun80(arr):
    if len(arr)<4:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = temp[-3:]
        return np.max(temp1)
    else:
        return 0
    
feature['threexMaxInterTime'] = grouped['actionTime'].apply(fun80)


#第八十一个值，时间间隔后两个的均值
def fun81(arr):
    if len(arr)<3:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = temp[-2:]
        return np.mean(temp1)
    else:
        return 0
    
feature['twoxMeanInterTime'] = grouped['actionTime'].apply(fun81)


#第八十二个值，时间间隔后两个的最小值
def fun82(arr):
    if len(arr)<3:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = temp[-2:]
        return np.min(temp1)
    else:
        return 0
    
feature['twoxMinInterTime'] = grouped['actionTime'].apply(fun82)


#第八十三个值，时间间隔后两个的最大值
def fun83(arr):
    if len(arr)<3:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = temp[-2:]
        return np.max(temp1)
    else:
        return 0
    
feature['twoxMaxInterTime'] = grouped['actionTime'].apply(fun83)


#第八十四个特征，时间间隔最小的四个的均值

def fun84(arr):
    if len(arr)<5:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = np.sort(temp)
        temp1 = temp1[:4]
        return np.mean(temp1)
    else:
        return 0
    
feature['fourMinMeanInterTime'] = grouped['actionTime'].apply(fun84)


#第八十五个特征，时间间隔最小的三个的均值

def fun85(arr):
    if len(arr)<4:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = np.sort(temp)
        temp1 = temp1[:3]
        return np.mean(temp1)
    else:
        return 0
    
feature['threeMinMeanInterTime'] = grouped['actionTime'].apply(fun85)


#第八十六个特征，时间间隔最小的两个的均值

def fun86(arr):
    if len(arr)<3:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = np.sort(temp)
        temp1 = temp1[:2]
        return np.mean(temp1)
    else:
        return 0
    
feature['twoxMinMeanInterTime'] = grouped['actionTime'].apply(fun86)


#第八十七个特征，时间间隔最小的五个的均值

def fun87(arr):
    if len(arr)<6:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = np.sort(temp)
        temp1 = temp1[:5]
        return np.mean(temp1)
    else:
        return 0
    
feature['fiverxMinMeanInterTime'] = grouped['actionTime'].apply(fun87)


#第八十八个特征，时间间隔最小的四个的标准差
def fun88(arr):
    if len(arr)<5:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = np.sort(temp)
        temp1 = temp1[:4]
        return np.std(temp1)
    else:
        return 0

feature['fourMinStdInterTime'] = grouped['actionTime'].apply(fun88)


#第八十九个特征，时间间隔最小的三个的标准差
def fun89(arr):
    if len(arr)<4:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = np.sort(temp)
        temp1 = temp1[:3]
        return np.std(temp1)
    else:
        return 0

feature['threeMinStdInterTime'] = grouped['actionTime'].apply(fun89)


#第九十个特征，时间间隔最小的5个的标准差
def fun90(arr):
    if len(arr)<6:
        return 0
    temp = []
    arr1 = np.array(arr)
    if len(arr1)>0:
        for i in range(len(arr1)-1):
            a = arr1[i+1] - arr1[i]
            temp.append(a)
        temp1 = np.sort(temp)
        temp1 = temp1[:5]
        return np.std(temp1)
    else:
        return 0


feature['twoxMinStdxxInterTime'] = grouped['actionTime'].apply(fun90)


#第九十一个特征，离最近的9的时间间隔均值乘以方差
feature['nineAndNine'] = feature['disNineVar']*feature['disNineMean']


#第九十二个特征，离最近的2的时间间隔均值
def fun92(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    
    m = 0
    temp_list = []
    if 2 not in temp1['actionType'].values:
         return 0
    else:
         for i in range(len(temp1)-1):
              if temp1[i:i+1]['actionType'].values != 2:
                   m = temp1[i+1:i+2]['actionTime'].values - temp1[i:i+1]['actionTime'].values
                   temp_list.append(m)
              else:
                   #print(temp1[i:i+1]['actionTime'].values)
                   break
         if len(temp_list)>0:
              temp_list = np.array(temp_list)
              #print(abs(np.mean(temp_list)))
              return abs(np.mean(temp_list))
         else:
              return 0
         
feature['disTwooVar'] = grouped.apply(fun92)


#第九十三个特征，离最近的3的时间间隔均值
def fun93(df,columns = 'actionTime'):
    temp1 = df.sort_index(by=columns, ascending=False)  #按时间倒序
    
    m = 0
    temp_list = []
    if 3 not in temp1['actionType'].values:
         return 0
    else:
         for i in range(len(temp1)-1):
              if temp1[i:i+1]['actionType'].values != 3:
                   m = temp1[i+1:i+2]['actionTime'].values - temp1[i:i+1]['actionTime'].values
                   temp_list.append(m)
              else:
                   #print(temp1[i:i+1]['actionTime'].values)
                   break
         if len(temp_list)>0:
              temp_list = np.array(temp_list)
              #print(abs(np.mean(temp_list)))
              return abs(np.mean(temp_list))
         else:
              return 0
         
feature['disThreeeVar'] = grouped.apply(fun93)


#第九十四个特征，离最近的3的距离
def fun94(arr):
     arr1 = np.array(arr)
     count = 1
     if 3 not in arr1:
          return 0
     for i in arr1[::-1]:
          if i != 3:
               count+=1
          else:
               return count
     
feature['lastTTHredtDis'] = grouped['actionType'].apply(fun94)


#第九十五个特征，用户的年龄
userProfile_train = pd.read_csv(r'E:\data\黄包车比赛\皇包车比赛\皇包车比赛数据-非压缩包\trainingset\userProfile_train.csv')
userProfile_test = pd.read_csv(r'E:\data\黄包车比赛\皇包车比赛\皇包车比赛数据-非压缩包\test\userProfile_test.csv')
userProfile_df = pd.concat((userProfile_train, userProfile_test),axis=0)

userProfile_df.drop('gender', axis=1, inplace=True)

feature = pd.merge(feature, userProfile_df, on='userid', how='left')




#第九十六个特征，之前是否订过精品，订过为1，未订过为0
orderHistory_train = pd.read_csv(r'E:\data\黄包车比赛\皇包车比赛\皇包车比赛数据-非压缩包\trainingset\orderHistory_train.csv')
orderHistory_test = pd.read_csv(r'E:\data\黄包车比赛\皇包车比赛\皇包车比赛数据-非压缩包\test\orderHistory_test.csv')
orderHistory_df = pd.concat((orderHistory_train, orderHistory_test), axis=0)
orderHistory_grouped = orderHistory_df.groupby('userid')

def fun97(arr):
    if 1 in arr['orderType'].values:
        return 1
    else:
        return 0

feature['HisorderType'] = orderHistory_grouped.apply(fun97)




feature.to_csv(r'E:\data\黄包车比赛\feature84.csv',index = False)


feature = pd.read_csv(r'E:\data\黄包车比赛\feature101.csv')

#九十七个特征,第一个为6，后面为7或者8的比例
def fun98(arr):
     arr1 = np.array(arr)
     if len(arr) == 1:
          return 0
     count = 0
     for i in range(len(arr1)-1):
          if arr1[i]==6 and (arr1[i+1]==8 or arr1[i+1]==7):
               count+=1
     return count/(len(arr1)-1)

feature['sixSevEigandSum'] = grouped['actionType'].apply(fun98)


#第九十八个特征，是否含有6,7,8,9的组合
def fun99(arr):
     arr1 = np.array(arr)
     if len(arr) < 4:
          return 0
     for i in range(len(arr1)-3):
          if (arr1[i] ==6 and arr1[i+1]==7 and arr1[i+2]==8 and arr[i+3]==9):
               return 1
     return 0

feature['sixSevEigNine'] = grouped['actionType'].apply(fun99)



feature = pd.read_csv(r'E:\data\黄包车比赛\皇包车比赛\feature70.csv')
feature.to_csv(r'E:\data\黄包车比赛\feature101.csv', index=False)


#第九十九个特征，6,8连用占总操作的比例
def fun99(arr):
     arr1 = np.array(arr)
     if len(arr) == 1:
          return 0
     count = 0
     for i in range(len(arr1)-1):
          if arr1[i]==6 and arr1[i+1]==8:
               count+=1
     return count/(len(arr1)-1)

feature['sixxAndEIGHT'] = grouped['actionType'].apply(fun99)


#第一百个特征，6,7,8连用的比例
def fun100(arr):
     arr1 = np.array(arr)
     if len(arr1) < 3:
          return 0
     count = 0
     for i in range(len(arr1)-2):
          if (arr1[i]==6 and arr1[i+1]==7 and arr1[i+2] == 8 ):
               count+=1
     return count/(len(arr1)-2)

feature['sixxAndSEvenEIGHT'] = grouped['actionType'].apply(fun100)


#第一百一个特征，最后一个6与最后一个7或者8（取最小值）的时间间隔
a = feature['disSix'] - feature['disSeven']
b = feature['disSix'] - feature['disEight']
temp = []
a1 = np.abs(a.values)
b1 = np.abs(b.values)
for i in range(len(a1)):
    if a1[i] > b1[i]:
        temp.append(a1[i])
    else:
        temp.append(b1[i])
m = pd.Series(temp,name='col')
feature = feature.set_index( m.index)
feature['lasiSixewiEight'] = m
feature = feature.set_index(feature.userid)



#保持索引和用户id一致
feature = feature.set_index(feature.userid)











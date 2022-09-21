# %%
#SiO2 Pbo  K20 sro fe2o3
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import GridSearchCV 
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier as ada 
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF 
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier as RF
df=pd.read_excel("table2_3.xlsx")
remove_list=['氧化钠(Na2O)','氧化镁(MgO)','氧化铝(Al2O3)','氧化铜(CuO)','五氧化二磷(P2O5)', '氧化锡(SnO2)','二氧化硫(SO2)']
df=df.drop(remove_list,axis=1)
df=df[df['未风化']==1]
df=df.drop(['未风化','风化'],axis=1)
x=df.iloc[:,1:-1].values
y=df.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=36,test_size=0.2)
transformer = StandardScaler()
x=transformer.fit_transform(x)
x_train=transformer.transform(x_train)
x_test=transformer.transform(x_test)

LR1=LogisticRegression( 
            C=0.1, class_weight=None, dual=False, fit_intercept=True,
            intercept_scaling=1,l1_ratio=None, max_iter=40, multi_class='auto', n_jobs=None,
            penalty='l2', random_state=40, solver='newton-cg', tol=0.0001,
            verbose=0, warm_start=False
           )

clf,label = LR1,"LR"
clf.fit(x_train, y_train)
w = clf.coef_                                # 模型系数(对应归一化数据)
b = clf.intercept_ 
y_predict=clf.predict(x_test)
print('y_test:',y_test)
print('y_pred:',y_predict)
print('{}在测试集模型上的准确率为：\n'.format(label),metrics.accuracy_score(y_test,y_predict))   
print("\n------模型参数-------")     
print( "模型系数:",w)
print( "模型阈值:",b)

# %%
print(LR1.predict(x_test))
x=df.iloc[:,1:-1].values
y=df.iloc[:,-1].values
accs_uw=[]
shakes=[0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
for i in shakes:
    x_shake=[]
    _,x_test,_,y_test=train_test_split(x,y,test_size=0.5)
    x_test=transformer.transform(x_test)
    for xs in x_test:
        x_shake.append([xt*(1+i*(random.random()-0.5)) for xt in xs])
    x_shake=np.array(x_shake)
    y_predict=LR1.predict(x_shake)
    print('y_test:',y_test)
    print('y_pred:',y_predict)
    acc=metrics.accuracy_score(y_test,y_predict)
    print('{}在测试集模型上的准确率为：\n'.format(label)) 
    accs_uw.append(acc)

# %%
df=pd.read_excel("table2_3.xlsx")
df=df[df['风化']==1]
df=df.drop(['未风化','风化'],axis=1)
df=df.drop(remove_list,axis=1)
x=df.iloc[:,1:-1].values
y=df.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=30,test_size=0.2)
transformer = StandardScaler()
x=transformer.fit_transform(x)
x_train=transformer.transform(x_train)
x_test=transformer.transform(x_test)

LR2=LogisticRegression( 
            C=0.1, class_weight=None, dual=False, fit_intercept=True,
            intercept_scaling=1,l1_ratio=None, max_iter=40, multi_class='auto', n_jobs=None,
            penalty='l2', random_state=40, solver='newton-cg', tol=0.0001,
            verbose=0, warm_start=False
           )

clf,label = LR2,"LR"
clf.fit(x_train, y_train)
w = clf.coef_                                # 模型系数(对应归一化数据)
b = clf.intercept_ 
y_predict=clf.predict(x_test)
print('y_test:',y_test)
print('y_pred:',y_predict)
print('{}在测试集模型上的准确率为：\n'.format(label),metrics.accuracy_score(y_test,y_predict))   
print("\n------模型参数-------")     
print( "模型系数:",w)
print( "模型阈值:",b)

# %%
accs_w=[]
x=df.iloc[:,1:-1].values
y=df.iloc[:,-1].values
for i in shakes:
    x_shake=[]
    _,x_test,_,y_test=train_test_split(x,y,test_size=0.5)
    x_test=transformer.transform(x_test)
    for xs in x_test:
        x_shake.append([xt*(1+i*(random.random()-0.5)) for xt in xs])
    x_shake=np.array(x_shake)
    y_predict=clf.predict(x_shake)
    print('y_test:',y_test)
    print('y_pred:',y_predict)
    acc=metrics.accuracy_score(y_test,y_predict)
    print('{}在测试集模型上的准确率为：\n'.format(label),acc) 
    accs_w.append(acc)

# %%
plt.style.use('ggplot')
fig,axes=plt.subplots(1,2,figsize=(14,6))
axes[0].plot(range(len(shakes)),accs_uw,label='未风化')
axes[1].plot(range(len(shakes)),accs_w,label='风化')
axes[0].set_xticks(range(len(shakes)),shakes)
axes[1].set_xticks(range(len(shakes)),shakes)
axes[0].set_title('未风化LR1')
axes[1].set_title('风化LR2')
axes[0].set_xlabel('扰动百分比范围')
axes[0].set_ylabel('准确率')

axes[1].set_xlabel('扰动百分比范围')
axes[1].set_ylabel('准确率')

plt.savefig('shake.png',dpi=300)
plt.show()




# %%
df3=pd.read_excel("附件.xlsx",sheet_name=2).fillna(0)
df3=df3.drop(remove_list,axis=1)
tem=df3.pop('表面风化')
ans=[]
arr=np.array(tem)
for i in range(8):
    x=df3.iloc[i,1:]
    if arr[i]=='无风化':
        pre_y=LR1.predict(x.values.reshape(1,-1))
    else:
        pre_y=LR2.predict(x.values.reshape(1,-1))
    ans.append(pre_y[0])

# %%
df3.insert(df3.shape[1],'玻璃类型',ans)
df3.loc[df3['玻璃类型']==0,'玻璃类型'] = '高钾'
df3.loc[df3['玻璃类型']==1,'玻璃类型'] = '铅钡'
df3.insert(1,'表面风化',tem)
df3

# %%
df4=pd.concat([df3[['文物编号']],df3[['玻璃类型']]],axis=1)
df4.to_excel('问题3预测结果.xlsx',index=False)



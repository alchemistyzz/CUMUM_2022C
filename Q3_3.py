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
from sklearn.ensemble import RandomForestClassifier as RF
df=pd.read_excel("table2_3.xlsx")
remove_list=['氧化钠(Na2O)','氧化镁(MgO)','氧化铝(Al2O3)','氧化铜(CuO)','五氧化二磷(P2O5)', '氧化锡(SnO2)','二氧化硫(SO2)']
df=df.drop(remove_list,axis=1)
df=df[df['未风化']==1]
df=df.drop(['未风化','风化'],axis=1)
# labels=["二氧化硅(SiO2)","氧化钾(K2O)","氧化铅(PbO)","氧化钡(BaO)",'玻璃类型']
# df=df[labels]
# df['玻璃类型'] = df['玻璃类型'].map({'高钾':0,'铅钡':1})
x=df.iloc[:,1:-1].values
y=df.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=30,test_size=0.2)
transformer = StandardScaler()
x=transformer.fit_transform(x)
x_train=transformer.transform(x_train)
x_test=transformer.transform(x_test)

# %%
accurary_uw=[]
std_uw=[]
iter_list=list(range(1,21))
iter_list=iter_list+[40,60,100,200]
print(type(iter_list))
for i in iter_list:
    LR1=LogisticRegression( 
            C=0.1, class_weight=None, dual=False, fit_intercept=True,
            intercept_scaling=1,l1_ratio=None, max_iter=i, multi_class='auto', n_jobs=None,
            penalty='l2', random_state=40, solver='newton-cg', tol=0.0001,
            verbose=0, warm_start=False
           )

    clf,label = LR1,"LR"
    scores = cross_val_score(clf, x, y, cv=5)
    print('{}在交叉验证准确率为：\n'.format(label),scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    accurary_uw.append(scores.mean())
    std_uw.append(scores.std() * 2)


# %%
plt.style.use('ggplot')
fig,axes=plt.subplots(1,2,figsize=(14,6))
axes[0].plot(range(len(accurary_uw)),accurary_uw,label='Accuracy')
axes[0].set_xlabel('max_iter')
axes[0].set_ylabel('Accuracy')
axes[0].set_xticks(range(len(accurary_uw)),[str(x) for x in iter_list])

axes[1].plot(range(len(accurary_uw)),std_uw,label='std')
axes[1].set_xlabel('max_iter')
axes[1].set_ylabel('std')
axes[1].set_xticks(range(len(accurary_uw)),[str(x) for x in iter_list])

plt.savefig('max_iter_uw.png')

# %%
df=pd.read_excel("table2_3.xlsx")
remove_list=['氧化钠(Na2O)','氧化镁(MgO)','氧化铝(Al2O3)','氧化铜(CuO)','五氧化二磷(P2O5)', '氧化锡(SnO2)','二氧化硫(SO2)']
df=df.drop(remove_list,axis=1)
df=df[df['风化']==1]
df=df.drop(['未风化','风化'],axis=1)
x=df.iloc[:,1:-1].values
y=df.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=30,test_size=0.2)
transformer = StandardScaler()
x=transformer.fit_transform(x)
x_train=transformer.transform(x_train)
x_test=transformer.transform(x_test)

# %%
accurary_uw=[]
std_uw=[]
iter_list=list(range(1,21))
iter_list=iter_list+[40,60,100,200]
print(type(iter_list))
for i in iter_list:
    LR2=LogisticRegression( 
            C=0.1, class_weight=None, dual=False, fit_intercept=True,
            intercept_scaling=1,l1_ratio=None, max_iter=i, multi_class='auto', n_jobs=None,
            penalty='l2', random_state=40, solver='newton-cg', tol=0.0001,
            verbose=0, warm_start=False
           )

    clf,label = LR2,"LR"
    scores = cross_val_score(clf, x, y, cv=5)
    print('{}在交叉验证准确率为：\n'.format(label),scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    accurary_uw.append(scores.mean())
    std_uw.append(scores.std() * 2)

# %%
plt.style.use('ggplot')
fig,axes=plt.subplots(1,2,figsize=(14,6))
axes[0].plot(range(len(accurary_uw)),accurary_uw,label='Accuracy')
axes[0].set_xlabel('max_iter')
axes[0].set_ylabel('Accuracy')
axes[0].set_xticks(range(len(accurary_uw)),[str(x) for x in iter_list])

axes[1].plot(range(len(accurary_uw)),std_uw,label='std')
axes[1].set_xlabel('max_iter')
axes[1].set_ylabel('std')
axes[1].set_xticks(range(len(accurary_uw)),[str(x) for x in iter_list])

plt.savefig('max_iter_w.png')

# %%
df=pd.read_excel("table2_3.xlsx")
remove_list=['氧化钠(Na2O)','氧化镁(MgO)','氧化铝(Al2O3)','氧化铜(CuO)','五氧化二磷(P2O5)', '氧化锡(SnO2)','二氧化硫(SO2)']
df=df.drop(remove_list,axis=1)
df=df[df['未风化']==1]
df=df.drop(['未风化','风化'],axis=1)
x=df.iloc[:,1:-1].values
y=df.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=30,test_size=0.2)
transformer = StandardScaler()
x=transformer.fit_transform(x)
x_train=transformer.transform(x_train)
x_test=transformer.transform(x_test)



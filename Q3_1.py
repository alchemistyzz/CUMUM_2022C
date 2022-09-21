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
import matplotlib as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier as RF
df=pd.read_excel("table2_3.xlsx")
df=df[df['未风化']==1]
df=df.drop(['未风化','风化'],axis=1)
x=df.iloc[:,1:-1].values
y=df.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=30,test_size=0.2)
transformer = StandardScaler()
x=transformer.fit_transform(x)
x_train=transformer.transform(x_train)
x_test=transformer.transform(x_test)

LR1=LogisticRegression( 
            C=0.1, class_weight=None, dual=False, fit_intercept=True,
            intercept_scaling=1,l1_ratio=None, max_iter=500, multi_class='auto', n_jobs=None,
            penalty='l2', random_state=30, solver='newton-cg', tol=0.0001,
            verbose=0, warm_start=False
           )

svc1=SVC(C=0.1, kernel='linear', degree=3, gamma='auto', coef0=0.0, 
    shrinking=True, probability=True, tol=0.001, cache_size=200,
    class_weight=None, verbose=False, max_iter=100, decision_function_shape='ovr', random_state=None)
for clf,label in zip([LR1,svc1],['LR','SVC']):
    # kfold = KFold(n_splits=5)

    clf.fit(x_train, y_train)
    w = clf.coef_                                # 模型系数(对应归一化数据)
    b = clf.intercept_                              # 模型阈值(对应归一化数据)
    y_predict=clf.predict(x_test)
    print('y_test:',y_test)
    print('y_pred:',y_predict)
    print('{}在测试集模型上的准确率为：\n'.format(label),metrics.accuracy_score(y_test,y_predict))
    print('{}在测试集模型上的召回率为：\n'.format(label),metrics.precision_score(y_test,y_predict))
    print('{}在训练集模型上的准确率为：\n'.format(label),metrics.accuracy_score(y_train,clf.predict(x_train)))
    print('{}在训练集模型上的准确率为：\n'.format(label),metrics.precision_score(y_train,clf.predict(x_train)))
    print('{}在综合准确率为：\n'.format(label),metrics.accuracy_score(y,clf.predict(x)))
    scores = cross_val_score(clf, x, y, cv=5)
    print('{}在交叉验证准确率为：\n'.format(label),scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    print("\n------模型参数-------")     
    print( "模型系数:",w)
    print( "模型阈值:",b)
    print('{}的AUC为：'.format(label),roc_auc_score(y, clf.predict(x)))

# %%
df=pd.read_excel("table2_3.xlsx")
df=df[df['风化']==1]
df=df.drop(['未风化','风化'],axis=1)
x=df.iloc[:,1:-1].values
y=df.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=30,test_size=0.2)
transformer = StandardScaler()
x=transformer.fit_transform(x)
x_train=transformer.transform(x_train)
x_test=transformer.transform(x_test)

LR2=LogisticRegression( 
            C=0.1, class_weight=None, dual=False, fit_intercept=True,
            intercept_scaling=1,l1_ratio=None, max_iter=500, multi_class='auto', n_jobs=None,
            penalty='l2', random_state=30, solver='newton-cg', tol=0.0001,
            verbose=0, warm_start=False
           )

svc2=SVC(C=0.1, kernel='linear', degree=3, gamma='auto', coef0=0.0, 
    shrinking=True, probability=True, tol=0.001, cache_size=200,
    class_weight=None, verbose=False, max_iter=100, decision_function_shape='ovr', random_state=None)

rf2=RF(n_estimators= 60, max_depth=13, min_samples_split=120,
                                  min_samples_leaf=20,max_features=7 ,oob_score=True, random_state=10)

for clf,label in zip([LR2,svc2],['LR','SVC']):
    # kfold = KFold(n_splits=5)

    clf.fit(x_train, y_train)
    w = clf.coef_                                # 模型系数(对应归一化数据)
    b = clf.intercept_                              # 模型阈值(对应归一化数据)
    y_predict=clf.predict(x_test)
    print('y_test:',y_test)
    print('y_pred:',y_predict)
    print('{}在测试集模型上的准确率为：\n'.format(label),metrics.accuracy_score(y_test,y_predict))
    print('{}在测试集模型上的召回率为：\n'.format(label),metrics.precision_score(y_test,y_predict))
    print('{}在训练集模型上的准确率为：\n'.format(label),metrics.accuracy_score(y_train,clf.predict(x_train)))
    print('{}在训练集模型上的准确率为：\n'.format(label),metrics.precision_score(y_train,clf.predict(x_train)))
    print('{}在综合准确率为：\n'.format(label),metrics.accuracy_score(y,clf.predict(x)))
    scores = cross_val_score(clf, x, y, cv=5)
    print('{}在交叉验证准确率为：\n'.format(label),scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    print("\n------模型参数-------")     
    print( "模型系数:",w)
    print( "模型阈值:",b)
    print('{}的AUC为：'.format(label),roc_auc_score(y,clf.predict(x)))



# %%
import pandas as pd

df=pd.read_excel("table2_1.xlsx")# 修改输入表格的类型
df['玻璃类型'] = df['玻璃类型'].map({'高钾':0,'铅钡':1}) #01标签映射
x=df.iloc[:,1:-3].values#训练的特征
print('训练特征为：',[column for column in df.iloc[:,1:-3]])
y=df.iloc[:,-1].values#训练标签

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
import pandas as pd

df=pd.read_excel("table2_1.xlsx")# 修改输入表格的类型
# labels=["二氧化硅(SiO2)","氧化钾(K2O)","氧化铅(PbO)","氧化钡(BaO)",'玻璃类型']
# df=df[labels]
df['玻璃类型'] = df['玻璃类型'].map({'高钾':0,'铅钡':1}) #01标签映射
x=df.iloc[:,1:-3].values#训练的特征
print('训练特征为：',[column for column in df.iloc[:,1:-3]])
y=df.iloc[:,-1].values#训练标签

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=30,test_size=0.3)
transformer = StandardScaler()
x=transformer.fit_transform(x)
x_train=transformer.transform(x_train)
x_test=transformer.transform(x_test)

#五种机器学习回归算法训练模型
LR=LogisticRegression(random_state=30)
svc=SVC(kernel='linear',random_state=30)
Ada=ada(random_state=30)
GBDT=GradientBoostingClassifier(random_state=30) 
rf=RF(random_state=30)

for clf,label in zip([LR,svc,Ada,GBDT,rf],['LR','SVC','Ada','GBDT','RF']):
    clf.fit(x_train, y_train)
    if(label=='LR'or label=='SVC'):
        w = clf.coef_                                # 模型系数(对应归一化数据)
        b = clf.intercept_                              # 模型阈值(对应归一化数据)
        print("\n------{}模型参数-------".format(label))     
        print( "模型系数:",w)
        print( "模型阈值:",b)
    y_predict=LR.predict(x_test)
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
    
    
    # print('{}的AUC为：'.format(label),roc_auc_score(y,LR.predict(x)))

# %%
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pylab import mpl
import matplotlib as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# plt.figure(dpi=300,figsize=(24,8))
plt.rcParams['axes.unicode_minus']=False
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

df=pd.read_excel("table2_1.xlsx")
# labels=["二氧化硅(SiO2)","氧化钾(K2O)","氧化铅(PbO)","氧化钡(BaO)",'玻璃类型']
# df=df[labels]
df['玻璃类型'] = df['玻璃类型'].map({'高钾':0,'铅钡':1})
x=df.iloc[:,1:-1].values
y=df.iloc[:,-1].values
d=df.corr(method="pearson")

plt.subplots(figsize = (12,12))
sns.heatmap(d,annot = True,vmax = 1,square = True,cmap = "Reds")
plt.rcParams['axes.facecolor']='snow'
plt.savefig('1-1 heatmap-pearson.png')
plt.show()



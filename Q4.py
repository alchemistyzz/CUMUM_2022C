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


df_wei_K=pd.read_excel("4.xlsx",sheet_name='未风化高钾')
d=df_wei_K.corr(method="pearson")
plt.subplots(figsize = (12,12))
sns.heatmap(d,annot = True,vmax = 1,square = True,cmap = "Reds")
plt.rcParams['axes.facecolor']='snow'
plt.title('未风化高钾')
plt.savefig('4 heatmap-pearson-wei-K.png')
plt.show()

df_wei_Ba=pd.read_excel("4.xlsx",sheet_name='未风化铅钡')
d=df_wei_Ba.corr(method="pearson")
plt.subplots(figsize = (12,12))
sns.heatmap(d,annot = True,vmax = 1,square = True,cmap = "Reds")
plt.rcParams['axes.facecolor']='snow'
plt.title('未风化铅钡')
plt.savefig('4 heatmap-pearson-wei-Ba.png')
plt.show()


df_yi_K=pd.read_excel("4.xlsx",sheet_name='风化高钾')
plt.subplots(figsize = (12,12))
d=df_yi_K.corr(method="pearson")
sns.heatmap(d,annot = True,vmax = 1,square = True,cmap = "Reds")
plt.rcParams['axes.facecolor']='snow'
plt.title('风化高钾')
plt.savefig('4 heatmap-pearson-yi-K.png')
plt.show()

df_yi_Ba=pd.read_excel("4.xlsx",sheet_name='风化铅钡')
plt.subplots(figsize = (12,12))
d=df_wei_Ba.corr(method="pearson")
sns.heatmap(d,annot = True,vmax = 1,square = True,cmap = "Reds")
plt.rcParams['axes.facecolor']='snow'
plt.title('风化铅钡')
plt.savefig('4 heatmap-pearson-yi-Ba.png')
plt.show()




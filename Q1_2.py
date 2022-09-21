# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
df1 = pd.read_excel("附件.xlsx",sheet_name=0)
df2 = pd.read_excel("sheet2_valid.xlsx")

# %%
conditions=[]
ids=[]
cds_labels=['未风化','一般风化','严重风化','高钾','铅钡']
for t in df2.iterrows():
    name=t[1].to_list()[0]
    cd=[0]*5
    ids.append(name)
    if name.find("未风化")!=-1:
        print("找到未风化点",name)
        cd[0]=1
        id=str(int(name[:2]))
        dd=np.array(df1[df1.文物编号.astype(str).str.contains(id)].iloc[0])
        if dd[2]=='高钾':
            cd[3]=1
        elif dd[2]=='铅钡':
            cd[4]=1
        conditions.append(cd)
    elif name.find("严重风化")!=-1:
        print("找到严重风化点",name)
        cd[2]=1
        id=str(int(name[:2]))
        dd=np.array(df1[df1.文物编号.astype(str).str.contains(id)].iloc[0])
        if dd[2]=='高钾':
            cd[3]=1
        elif dd[2]=='铅钡':
            cd[4]=1
        conditions.append(cd)
    else:
        
        id=str(int(name[:2]))
        dd=np.array(df1[df1.文物编号.astype(str).str.contains(id)].iloc[0])
        if dd[4]=='无风化':
            cd[0]=1
        else:
            cd[1]=1
        if dd[2]=='高钾':
            cd[3]=1
        elif dd[2]=='铅钡':
            cd[4]=1
        conditions.append(cd)


# %%
dc1=pd.concat([pd.DataFrame(ids,columns=['文物采样点']),pd.DataFrame(conditions,columns=cds_labels)],axis=1)
dc1.to_excel("conditions1.xlsx",index=False)

# %%
cds_labels=['未风化','一般风化','严重风化','高钾','铅钡','bias']

# %%
import geatpy as ea
import math

# %%
vars_excel=[]
elements=[]
ObjV=[]

for i in range(1,15):
    elements.append(df2.columns[i])
    values=df2.iloc[:,i].to_numpy().reshape(-1)
    maxv=max(values)
    minv=max(min(values),0)

    def evalVars(Vars):
        lost=[]
        const=[]
        for Var in Vars:
            l=0
            c=[]
            for i in range(len(conditions)):
                kVar=Var[0:5]
                bVar=Var[5]
                l+=(kVar.dot(conditions[i])+bVar-values[i])**2
                c.append(kVar.dot(conditions[i])+bVar-maxv)
                c.append(minv-kVar.dot(conditions[i])-bVar)
                c.append(-(min(kVar[0:3])+min(kVar[3:5])+bVar))
            l=l/len(conditions)
            l=math.sqrt(l)
            const.append(c)
            lost.append(l)

        ObjV=np.array(lost).reshape(-1,1)
        CV = np.array(const)

        return ObjV,CV

    problem = ea.Problem(name=df2.columns[i],
                        M=1,  # 目标维数
                        maxormins=[1],
                        Dim=6,  # 决策变量维数
                        varTypes=[0, 0, 0, 0, 0,0],  # 决策变量的类型
                        lb=[-100, -100, -100, -100, -100,-100],  # 决策变量下界
                        ub=[100, 100, 100, 100, 100,100],  # 决策变量上界
                        evalVars=evalVars)

    algorithm = ea.soea_SEGA_templet(problem,
                                    ea.Population(Encoding='RI', NIND=50),
                                    MAXGEN=200,  # 最大进化代数。
                                    logTras=1,  # 表示每隔多少代记录一次日志信息，0表示不记录。
                                    trappedValue=1e-6,  # 单目标优化陷入停滞的判断阈值。
                                    maxTrappedCount=20)  # 进化停滞计数器最大上限值。
# 求解
    res = ea.optimize(algorithm, seed=1, verbose=True, drawing=1, outputMsg=True, drawLog=False, saveFlag=True, dirName='result')
    
    
    vars_excel.append(res['Vars'].reshape(-1).tolist())
    ObjV.append(res['ObjV'].reshape(-1).tolist())
    print(res)



# %%
da=pd.DataFrame(elements,columns=['元素'])
db=pd.DataFrame(vars_excel,columns=cds_labels)
dc=pd.DataFrame(ObjV,columns=['平均标准差'])
dd=pd.concat([da,db,dc],axis=1)
dd.to_excel("result1_2.xlsx",index=False)



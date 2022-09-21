# %%
import math
import pandas as pd
import numpy as np

# %%
df2=pd.read_excel('sheet2_valid.xlsx')
cd_df= pd.read_excel('conditions1.xlsx').iloc[:,1:]
cd_result1=pd.read_excel('result1_2.xlsx').iloc[:,1:-1]

origins=df2.iloc[:,1:]
condition_array=np.array(cd_df)
result_array=np.array(cd_result1)
origins_array=np.array(origins)

# %%
origins_max=np.max(origins_array,axis=0)
origins_max

# %%
result2=[]
for i in range(len(origins_array)):
    elements=[0.]*14
    if condition_array[i][0]==1:
        result2.append(elements)
        continue
    bad=condition_array[i]
    print(bad)
    good=np.array([1,0,0]+list(condition_array[i])[3:])
    print(good)
    for j in range(result_array.shape[0]):
        b=np.dot(bad,result_array[j][0:5])+result_array[j][5]
        print("b",j,b)
        g=np.dot(good,result_array[j][0:5])+result_array[j][5]
        print("g",j,g)
        print("g/b",j,g/b)
        if b==0:
            elements[j]=g
        else:
            if (g/b)*origins_array[i][j]>origins_max[j]:
                elements[j]=g
            else:
                elements[j]=(g/b)*origins_array[i][j]
    se=sum(elements)
    elements=[e*100/se for e in elements]
    result2.append(elements)

# %%
da=pd.DataFrame(result2,columns=origins.columns)
dr=pd.concat([df2.iloc[:,0],da],axis=1)
dr.to_excel('result2.xlsx',index=False)
dr[dr.iloc[:,1:].sum(axis=1)>0].to_excel('result2_2_cut.xlsx',index=False)
df2[dr.iloc[:,1:].sum(axis=1)>0].to_excel('result2_origin.xlsx',index=False)



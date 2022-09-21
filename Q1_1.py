# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# %%
df2=pd.read_excel("附件.xlsx",sheet_name=1)
df2_valid=df2[(df2.iloc[:,1:].sum(axis=1)<105)&(df2.iloc[:,1:].sum(axis=1)>85)]
d=df2[(df2.iloc[:,1:].sum(axis=1)<105)&(df2.iloc[:,1:].sum(axis=1)>85)]
df2.reset_index(drop=True,inplace=True)77
df2_valid.to_excel("sheet2_valid.xlsx",index=False)
df1=pd.read_excel("附件.xlsx",sheet_name=0)
df1=df1.fillna("杂色")

# %%
colors=["浅绿","绿",'深绿','蓝绿','浅蓝','深蓝','紫','黑','杂色']
shapes=['A','B','C']
types=['铅钡','高钾']
def diff_color(color):
    i=1
    for c in colors:
        if color == c:
            return i
        i=i+1
def diff_shape(t):
    if t=='A':
        return 1
    elif t=='B':
        return 2
    else:
        return 3
def diff_type(s):
    if s=="铅钡":
        return 1
    else:
        return 2

def diff(f):
    if f == '风化':
        return 1
    else:
        return 0

# %%
objects=[]
for a in df1.iterrows():
    a=a[1].tolist()
    objects.append((diff_shape(a[1]),diff_type(a[2]),diff_color(a[3]),diff(a[4])))

# %%
def show_shape(objects):
    w=[0,0,0]
    uw=[0,0,0]
    for o in objects:
        if(o[3]==1):
            w[o[0]-1]=w[o[0]-1]+1
        else:
            uw[o[0]-1]=uw[o[0]-1]+1
   
    sum=[w[0]+uw[0],w[1]+uw[1],w[2]+uw[2]] 
    print(sum)
    
    wp=np.true_divide(w,sum)
    uwp=np.true_divide(uw,sum)
    uwp=uwp
    plt.style.use('ggplot')
    fig,axes=plt.subplots(1,3,figsize=(10,4))
    axes[0].pie([wp[0],uwp[0]],labels=['风化','未风化'],autopct='%1.1f%%',textprops={'fontsize': 14})
    axes[0].set_title('A类花纹')
    axes[1].pie([wp[1],uwp[1]],labels=['风化','未风化'],autopct='%1.1f%%',textprops={'fontsize': 14})
    axes[1].set_title('B类花纹')
    axes[2].pie([wp[2],uwp[2]],labels=['风化','未风化'],autopct='%1.1f%%',textprops={'fontsize': 14})
    axes[2].set_title('C类花纹')
    plt.legend()
    plt.savefig("花纹类别pie.png",dpi=400)
    plt.show()

    

# %%
show_shape(objects)

# %%
from tkinter import font


def show_type(objects):
    w=[0,0]
    uw=[0,0]
    for o in objects:
        if(o[3]==1):
            w[o[1]-1]=w[o[1]-1]+1
        else:
            uw[o[1]-1]=uw[o[1]-1]+1
   
    sum=[w[0]+uw[0],w[1]+uw[1]] 
    # print(sum)
    
    wp=np.true_divide(w,sum)
    uwp=np.true_divide(uw,sum)
    plt.style.use('ggplot')
    fig,axes=plt.subplots(1,2,figsize=(10,4))

    axes[0].pie([wp[0],uwp[0]],labels=['风化','未风化'],autopct='%1.1f%%',textprops={'fontsize': 14})
    axes[0].set_title('铅钡类')
    axes[1].pie([wp[1],uwp[1]],labels=['风化','未风化'],autopct='%1.1f%%',textprops={'fontsize': 14})
    axes[1].set_title('高钾类')
    plt.legend(loc='lower right')
    plt.savefig("玻璃类别pie.png",dpi=400)
    plt.show()

# %%
show_type(objects)

# %%
def show_color(objects):
    w=[0]*9
    uw=[0]*9
    for o in objects:
        if(o[3]==1):
            w[o[2]-1]=w[o[2]-1]+1
        else:
            uw[o[2]-1]=uw[o[2]-1]+1
   
    sum=[w[i]+uw[i] for i in range(9)]
    
    wp=np.true_divide(w,sum)
    uwp=np.true_divide(uw,sum)
    plt.style.use('ggplot')
    fig,axes=plt.subplots(3,3,figsize=(12,12))
    for i in range(3):
        for j in range(3):
            axes[i,j].pie([wp[i*3+j],uwp[i*3+j]],labels=['风化','未风化'],autopct='%1.1f%%',textprops={'fontsize': 12})
            axes[i,j].set_title(colors[i*3+j])
    plt.legend(loc='upper right')
    plt.savefig("颜色类别pie.png",dpi=400)
    plt.show()

# %%
show_color(objects)

# %%
#计算信息熵
def cal_information_entropy(data):
    data_label = data.iloc[:,-1]
    label_class =data_label.value_counts() 
    Ent = 0
    for k in label_class.keys():
        p_k = label_class[k]/len(data_label)
        Ent += -p_k*np.log2(p_k)
    return Ent

def cal_information_gain(data, a):
    Ent = cal_information_entropy(data)
    feature_class = data[a].value_counts() 
    gain = 0
    for v in feature_class.keys():
        weight = feature_class[v]/data.shape[0]
        Ent_v = cal_information_entropy(data.loc[data[a] == v])
        gain += weight*Ent_v

    print(f'特征{a}的信息增益为{Ent - gain}')
    return Ent - gain

def cal_information_gain_continuous(data, a):
    n = len(data) 
    data_a_value = sorted(data[a].values) 
    Ent = cal_information_entropy(data) 
    select_points = []
    for i in range(n-1):
        val = (data_a_value[i] + data_a_value[i+1]) / 2 
        data_left = data.loc[data[a]<val]
        data_right = data.loc[data[a]>val]
        ent_left = cal_information_entropy(data_left)
        ent_right = cal_information_entropy(data_right)
        result = Ent - len(data_left)/n * ent_left - len(data_right)/n * ent_right
        select_points.append([val, result])
    select_points.sort(key = lambda x : x[1], reverse= True) 
    return select_points[0][0], select_points[0][1]

def get_most_label(data):
    data_label = data.iloc[:,-1]
    label_sort = data_label.value_counts(sort=True)
    return label_sort.keys()[0]

def get_best_feature(data):
    features = data.columns[:-1]
    res = {}
    for a in features:
        if a in continuous_features:
            temp_val, temp = cal_information_gain_continuous(data, a)
            res[a] = [temp_val, temp]
        else:
            temp = cal_information_gain(data, a)
            res[a] = [-1, temp] #离散值没有划分点，用-1代替

    res = sorted(res.items(),key=lambda x:x[1][1],reverse=True)
    print(f'最佳划分特征为{res[0][0]}')
    return res[0][0],res[0][1][0]

def drop_exist_feature(data, best_feature):
    attr = pd.unique(data[best_feature])
    new_data = [(nd, data[data[best_feature] == nd]) for nd in attr]
    new_data = [(n[0], n[1].drop([best_feature], axis=1)) for n in new_data]
    return new_data

def create_tree(data):
    data_label = data.iloc[:,-1]
    if len(data_label.value_counts()) == 1:
        return data_label.values[0]
    if all(len(data[i].value_counts()) == 1 for i in data.iloc[:,:-1].columns): 
        return get_most_label(data)
    best_feature, best_feature_val = get_best_feature(data) 
    if best_feature in continuous_features:
        node_name = best_feature + '<' + str(best_feature_val)
        Tree = {node_name:{}}
        Tree[node_name]['是'] = create_tree(data.loc[data[best_feature] < best_feature_val])
        Tree[node_name]['否'] = create_tree(data.loc[data[best_feature] > best_feature_val])
    else:
        Tree = {best_feature:{}}
        exist_vals = pd.unique(data[best_feature]) 
        if len(exist_vals) != len(column_count[best_feature]): 
            no_exist_attr = set(column_count[best_feature]) - set(exist_vals)
            for no_feat in no_exist_attr:
                Tree[best_feature][no_feat] = get_most_label(data)
        for item in drop_exist_feature(data, best_feature):
            print(f'当前特征为{best_feature},特征值为{item[0]}')
            Tree[best_feature][item[0]] = create_tree(item[1])
    return Tree

#根据创建的决策树进行分类
def predict(Tree , test_data):
    first_feature = list(Tree.keys())[0]
    if (feature_name:= first_feature.split('<')[0]) in continuous_features:
        second_dict = Tree[first_feature]
        val = float(first_feature.split('<')[-1])
        input_first = test_data.get(feature_name)
        if input_first < val:
            input_value = second_dict['是']
        else:
            input_value = second_dict['否']
    else:
        second_dict = Tree[first_feature]
        input_first = test_data.get(first_feature)
        input_value = second_dict[input_first]
    if isinstance(input_value , dict):
        class_label = predict(input_value, test_data)
    else:
        class_label = input_value
    return class_label

data = df1
data.reset_index(drop=True, inplace=True)
data=data.iloc[:,1:]
    # 统计每个特征的取值情况作为全局变量
column_count = dict([(ds, list(pd.unique(data[ds]))) for ds in data.iloc[:, :-1].columns])
continuous_features = []  #连续值
dicision_tree = create_tree(data)
print(dicision_tree)
test_data={'颜色':'蓝绿','纹饰':'C','类型':'高钾'}
result = predict(dicision_tree, test_data)
print(result)

# %%

def data_flatten(key,val,con_s='_',basic_types=(str,int,float,bool,complex,bytes)):
    if isinstance(val, dict):
        for ck,cv in val.items():
            yield from data_flatten(con_s.join([key,ck]).lstrip('_'), cv)
    elif isinstance(val, (list,tuple,set)):
        for item in val:
            yield from data_flatten(key,item)
    elif isinstance(val, basic_types) or val is None:
        yield str(key).lower(),val

for i in data_flatten('',dicision_tree):
    print(i)



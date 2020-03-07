#!/usr/bin/env python
# coding: utf-8

# In[2]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"#让所有不是在最后一行的单个变量也能print


# In[225]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family']='Times New Roman'
matplotlib.rcParams['font.sans-serif']=['Times New Roman']


# # 对方球员的weight from anotheripynb

# In[3]:


import pandas as pd


# In[4]:


passing_copy_weight_O=pd.read_csv('passing_feature_weight_0.csv')


# # 己方球员的weight

# In[6]:


full=pd.read_csv("fullevents.csv")


# In[7]:


full_copy=full.copy()


# In[8]:


passing=pd.read_csv("passingevents.csv")


# In[9]:


passing_copy=passing.copy()


# In[10]:


full_copy.head(10)


# In[11]:


full_copy['Position']=full['OriginPlayerID'].apply(lambda x:x.split("_")[-1][0])

passing_copy['Position']=passing['OriginPlayerID'].apply(lambda x:x.split("_")[-1][0])


# In[15]:


passing_copy_H=passing_copy[passing_copy['TeamID']=='Huskies']


# In[17]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit_transform(passing_copy_H['OriginPlayerID'])


# In[19]:


map_player_id={'Huskies_D1': 0,'Huskies_D10': 1,'Huskies_D2': 2,'Huskies_D3': 3,'Huskies_D4': 4,'Huskies_D5': 5,'Huskies_D6': 6,'Huskies_D7': 7,'Huskies_D8': 8,'Huskies_D9': 9,'Huskies_F1': 10,'Huskies_F2': 11,'Huskies_F3': 12,'Huskies_F4': 13,'Huskies_F5': 14,'Huskies_F6': 15,'Huskies_G1': 16,'Huskies_M1': 17,'Huskies_M10': 18,'Huskies_M11': 19,'Huskies_M12': 20,'Huskies_M13': 21,'Huskies_M2': 22,'Huskies_M3': 23,'Huskies_M4': 24,'Huskies_M5': 25,'Huskies_M6': 26,'Huskies_M7': 27,'Huskies_M8': 28,'Huskies_M9': 29}


map_player_id_O={'D1': 0,'D2': 1,'D3': 2,'D4': 3,'D5': 4,'D6': 5,'D7': 6,'F1': 7,'F2': 8,'F3': 9,'F4': 10,'F5': 11,'F6': 12,'G1': 13,'G2': 14,'M1': 15,'M2': 16,'M3': 17,'M4': 18,'M5': 19,'M6': 20,'M7': 21,'M8': 22}

# In[20]:


passing_copy_H['OriginPlayerIDFit']=preprocessing.LabelEncoder().fit_transform(passing_copy_H['OriginPlayerID'])


# ### 查看fit_transform是如何map的

# In[21]:


data = ['A', 'A', 'B', 'C', 'B', 'B'] # `y`

le = preprocessing.LabelEncoder()
mapped = le.fit_transform(data)

mapping = dict(zip(le.classes_, range(0, len(le.classes_)+1)))
mapped
print(mapping)
# {'A': 1, 'B': 2, 'C': 3}


# In[24]:


passing_copy_H['DestinationPlayerIDFit']=preprocessing.LabelEncoder().fit_transform(passing_copy_H['DestinationPlayerID'])


# In[25]:


def getpair(data):
    left=data['OriginPlayerIDFit']
    right=data['DestinationPlayerIDFit']
    return (left,right)

passing_copy_H['PlayerIDFitPair']=passing_copy_H.apply(getpair,axis=1)
passing_copy_H['PassingBetweenPlayerCount']=[passing_copy_H.groupby('PlayerIDFitPair')['PlayerIDFitPair'].count()[i] for i in passing_copy_H['PlayerIDFitPair']]


# In[34]:


passing_copy_H['DestinationPosition']=passing_copy_H['DestinationPlayerID'].apply(lambda x:x.split("_")[-1][0])


# In[36]:


def getpair(data):
    left=data['Position']
    right=data['DestinationPosition']
    return (left,right)

passing_copy_H['PositionFitPair']=passing_copy_H.apply(getpair,axis=1)


# In[37]:


passing_copy_H['PassingBetweenPositionCount']=[passing_copy_H.groupby('PositionFitPair')['PositionFitPair'].count()[i] for i in passing_copy_H['PositionFitPair']]


# In[39]:


passing_copy_H.to_csv('passing_feature.csv')


# In[40]:


PositionFitPairMap={('F','F'):1,('F','M'):0.75,('F','D'):0.5,('F','G'):0.25,('D','D'):1,('D','M'):0.75,('D','G'):0.5,('D','F'):0.5,('M','F'):0.25,('M','M'):1,('M','D'):0.25,('M','G'):0.75,('G','F'):0.125,('G','M'):0.25,('G','D'):0.5,('G','G'):0}


# In[41]:


alpha=1/3
beta=1/3
gamma=1/3


# In[42]:


passing_copy_H=pd.read_csv('passing_feature.csv')


# In[43]:


def dist(line):
    return pow(pow((line['EventOrigin_x']-line['EventDestination_x']),2)+pow((line['EventOrigin_y']-line['EventDestination_y']),2),1/2)


# In[44]:


passing_copy_H['dist']=passing_copy_H.apply(dist,axis=1)


# In[46]:


0 in passing_copy_H['dist'].unique()


# In[47]:


# passing_copy_H['dist'].value_counts()
from collections import Counter

result = Counter(passing_copy_H['dist'])
print(result[25.9])


# In[48]:


passing_copy_H[passing_copy_H['dist']<1]


# In[49]:


#PositionFitPairMap,dist,PassingBetweenPlayerCount


# In[50]:


def PositionFitPairScoreMap(line):
    return PositionFitPairMap[eval(line['PositionFitPair'])]


# In[51]:


passing_copy_H['PositionFitPairScore']=passing_copy_H.apply(PositionFitPairScoreMap,axis=1)


# In[53]:


passing_copy_H['OneDevideDist']=passing_copy_H.apply(lambda x:1.0/x['dist'] if x['dist']!=0 else 1,axis=1)


# In[56]:


import numpy as np


# In[57]:


passing_copy_H['StdPassingBetweenPlayerCount']=(passing_copy_H['PassingBetweenPlayerCount'] - passing_copy_H['PassingBetweenPlayerCount'].min())/(passing_copy_H['PassingBetweenPlayerCount'].max() - passing_copy_H['PassingBetweenPlayerCount'].min())


# In[59]:


def weight(line):
    return alpha*line['PositionFitPairScore']+beta*line['OneDevideDist']+gamma*line['StdPassingBetweenPlayerCount']


# In[60]:


passing_copy_H['weight']=passing_copy_H.apply(weight,axis=1)



# In[59]:


passing_copy_H.to_csv('passing_feature_weight.csv')


# In[62]:


passing_copy_weight=pd.read_csv('passing_feature_weight.csv')


# In[63]:


def get_avg(line):
    return passing_copy_weight.groupby('PlayerIDFitPair')['weight'].mean()[line['PlayerIDFitPair']]


# In[64]:


passing_copy_weight['weight']=passing_copy_weight.apply(get_avg,axis=1)



# In[67]:


passing_copy_weight.to_csv('passing_feature_weight_mean.csv')


# In[68]:


passing_copy_weight.groupby('PlayerIDFitPair')['weight'].mean().to_csv('player_weight.csv')


# In[69]:


passing_copy_weight2=pd.read_csv('player_weight.csv',header=None)


# In[70]:


def to_split(line):
    return eval(line[0])[0],eval(line[0])[1]


# In[71]:


passing_copy_weight2['left']=passing_copy_weight2.apply(to_split,axis=1).apply(lambda x:x[0])
passing_copy_weight2['right']=passing_copy_weight2.apply(to_split,axis=1).apply(lambda x:x[1])


# In[72]:


passing_copy_weight2['weight']=passing_copy_weight2[1]


# In[73]:


passing_copy_weight2[['left','right','weight']].to_csv('player_weight_2.csv',index=None)



# In[75]:


def OutCount(line):
    return passing_copy_weight['OriginPlayerID'].value_counts()[line['OriginPlayerID']]


# In[76]:


passing_copy_weight['OutCount']=passing_copy_weight.apply(OutCount,axis=1)


# In[77]:


def InCount(line):
    return passing_copy_weight['DestinationPlayerID'].value_counts()[line['DestinationPlayerID']]


# In[78]:


passing_copy_weight['InCount']=passing_copy_weight.apply(InCount,axis=1)



# In[87]:


first_half_data=passing_copy_weight[passing_copy_weight['MatchPeriod']=='1H']



# In[84]:


second_half_data=passing_copy_weight[passing_copy_weight['MatchPeriod']=='2H']


# In[223]:


first_half_data.to_csv('first_half_data.csv')


# In[85]:


def AvgPosition_x(line):
    return first_half_data.groupby('OriginPlayerID')['EventOrigin_x'].mean()[line['OriginPlayerID']]


# In[88]:


first_half_data['AvgPosition_x']=first_half_data.apply(AvgPosition_x,axis=1)


# In[89]:


def AvgPosition_y(line):
    return first_half_data.groupby('OriginPlayerID')['EventOrigin_y'].mean()[line['OriginPlayerID']]


# In[90]:


first_half_data['AvgPosition_y']=first_half_data.apply(AvgPosition_y,axis=1)


# In[91]:


def AvgWeight(line):
    return first_half_data.groupby('PlayerIDFitPair')['weight'].mean()[line['PlayerIDFitPair']]


# In[92]:


first_half_data['AvgWeight']=first_half_data.apply(AvgWeight,axis=1)


# ## OutAvgWeight,InAvgWeight

# In[93]:


def OutAvgWeight(line):
    return passing_copy_weight.groupby('OriginPlayerID')['weight'].mean()[line['OriginPlayerID']]
def InAvgWeight(line):
    return passing_copy_weight.groupby('DestinationPlayerID')['weight'].mean()[line['DestinationPlayerID']]


# In[94]:


passing_copy_weight['OutAvgWeight']=passing_copy_weight.apply(OutAvgWeight,axis=1)
passing_copy_weight['InAvgWeight']=passing_copy_weight.apply(InAvgWeight,axis=1)


# In[97]:


first_half_data_to_draw=first_half_data[['OriginPlayerID','DestinationPlayerID','OriginPlayerIDFit','DestinationPlayerIDFit','AvgWeight','AvgPosition_x','AvgPosition_y','OutCount','Position']].drop_duplicates(inplace=False)


# In[93]:


first_half_data_to_draw.to_csv('first_half_data_to_draw.csv')


# In[95]:


'Huskies_M2'.split('_')[1]


# In[98]:


def cut_Huskies(line):
    return line['OriginPlayerID'].split('_')[1],line['DestinationPlayerID'].split('_')[1]


# In[99]:


cut=first_half_data_to_draw.apply(cut_Huskies,axis=1)
first_half_data_to_draw['OriginPlayerID']=cut.apply(lambda x:x[0])
first_half_data_to_draw['DestinationPlayerID']=cut.apply(lambda x:x[1])


# In[101]:


positions=first_half_data_to_draw.drop_duplicates(subset=['OriginPlayerID'],keep='first',inplace=False)[['AvgPosition_x','AvgPosition_y','OriginPlayerID']]




# In[435]:


import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.image as img

def minard_graph():
    data_F = first_half_data_to_draw[first_half_data_to_draw['Position']=='F']
    data_M = first_half_data_to_draw[first_half_data_to_draw['Position']=='M']
    data_D = first_half_data_to_draw[first_half_data_to_draw['Position']=='D']
    data_G = first_half_data_to_draw[first_half_data_to_draw['Position']=='G']
#24.0,54.9,340000,A,1
    #data_pos=first_half_data_to_draw[first_half_data_to_draw['Position']=='G']
    c = {}
#     for line in positions.split('\n'):
#         x, y, name = line.split(',')
#         c[name] = (float(x), float(y))
    for index,row in positions.iterrows():
        c[row['OriginPlayerID']]=(float(row['AvgPosition_x']),float(row['AvgPosition_y']))
    g = []
    for data in [data_F, data_M, data_D, data_G]:
        G = nx.Graph()
        i = 0
        G.pos = {}  # location
        G.pop = {}  # size
        G.alpha = {}  # cuxi
        last=None

        for index,row in data.iterrows():
#OriginPlayerID	DestinationPlayerID	OriginPlayerIDFit	DestinationPlayerIDFit	
#AvgWeight	AvgPosition_x	AvgPosition_y	PassingBetweenPlayerCount	Position
            G.pos[i] = (float(row['AvgPosition_x']), float(row['AvgPosition_y']))
            G.pop[i] = float(row['OutCount'])*2
            G.alpha[i]=row['AvgWeight']
            #G.add_edge(row['OriginPlayerIDFit'],row['DestinationPlayerIDFit'])
            if last is None:
                last = i
            else:
                G.add_edge(i, last)
                last = i
            i = i + 1
        g.append(G)
    return g,c
if __name__ == "__main__":

    (g, player) = minard_graph()
    fig=plt.figure('court.jpg', figsize=(12*105/68.0,12))

    colors = ['#FBDB8E', '#FF6100', 'r','#FFFF00']
    for G in g:
        c = colors.pop(0)
        #print(G.pop)
        node_size = [G.pop[n] for n in G]
        width = [G.alpha[n] for n in G]
        width_next=width.pop(0)
        nx.draw_networkx_edges(G, G.pos, edge_color=c, width=2, alpha=width_next)
        nx.draw_networkx_nodes(G, G.pos, node_size=node_size, node_color=c, alpha=1)
        #nx.draw_networkx_nodes(G, G.pos, node_size=5, node_color='k')
    print(player)
    for c in player:
        x, y = player[c]
        plt.text(x, y , c,ha='center',color='#000000',fontsize=15)#city 位置的标记
    plt.show()
    plt.savefig("weight_first_half.png")

# In[325]:


from networkx.algorithms.community import kclique as kclique
klist = list(k_clique_communities(G,5))
print(klist)
nx.draw(G,pos =G.pos, with_labels=False)
nx.draw(G,pos = G.pos, nodelist = klist[0], node_color = 'b')
nx.draw(G,pos = G.pos, nodelist = klist[1], node_color = 'y')
plt.show()


# In[317]:


klist


# ### 草稿请跳过

# In[103]:


data_F = first_half_data_to_draw[first_half_data_to_draw['Position']=='F']
data_M = first_half_data_to_draw[first_half_data_to_draw['Position']=='M']
data_D = first_half_data_to_draw[first_half_data_to_draw['Position']=='D']
data_G = first_half_data_to_draw[first_half_data_to_draw['Position']=='G']




# In[104]:


kiin=passing_copy_weight['DestinationPlayerID'].value_counts()#kiin


# In[105]:


kiout=passing_copy_weight['OriginPlayerID'].value_counts()#kiout


# In[106]:


weightin=passing_copy_weight.groupby('OriginPlayerID')['InAvgWeight'].mean()#weightin
weightout=passing_copy_weight.groupby('OriginPlayerID')['OutAvgWeight'].mean()#weightout




# In[108]:


import math
math.isnan(weightin['Huskies_M7'])


# In[109]:


PlayerScore=[((kiout[i]*pow((weightout[i]/kiout[i]),0.8))+(kiin[i]*pow((weightin[i]/kiin[i]),0.8)))/2 if not math.isnan(weightin[i]) else 0 for i in passing_copy_weight['OriginPlayerID'].unique()]


# In[110]:


from sklearn.cluster import KMeans
import numpy as np
X = np.array(PlayerScore).reshape(-1,1)
kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
kmeans.labels_



# In[112]:


import sklearn.cluster as skc  # 密度聚类
dbscan=skc.DBSCAN(eps=0.1, min_samples=2).fit(X)
dbscan.labels_


# In[113]:


for i in range(len(passing_copy_weight['OriginPlayerID'].unique())):
    print(passing_copy_weight['OriginPlayerID'].unique()[i],dbscan.labels_[i])


# In[114]:


passing_copy_weight['OriginPlayerID'].unique(),kmeans.labels_


# In[115]:


for i in range(len(passing_copy_weight['OriginPlayerID'].unique())):
    print(passing_copy_weight['OriginPlayerID'].unique()[i],kmeans.labels_[i])


# In[ ]:





# ## 中心性介质betweenness 最短距离shortestpath

# In[335]:



import networkx as nx
import pylab
import numpy as np
#自定义网络
row=np.array([0,0,0,1,2,3,6])
col=np.array([1,2,3,4,5,6,7])
value=np.array([1,2,1,8,1,3,5])
 
print('生成一个空的有向图')
G=nx.DiGraph()
print('为这个网络添加节点...')
for i in range(0,np.size(col)+1):
    G.add_node(i)
print('在网络中添加带权中的边...')
for i in range(np.size(row)):
    G.add_weighted_edges_from([(row[i],col[i],value[i])])
 
print('给网路设置布局...')
pos=nx.shell_layout(G)
print('画出网络图像：')
nx.draw(G,pos,with_labels=True, node_color='white', edge_color='red', node_size=400, alpha=0.5 )
pylab.title('Self_Define Net',fontsize=15)
pylab.show()
 
 
'''
Shortest Path with dijkstra_path
'''
print('dijkstra方法寻找最短路径：')
path=nx.dijkstra_path(G, source=0, target=7)
print('节点0到7的路径：', path)
print('dijkstra方法寻找最短距离：')
distance=nx.dijkstra_path_length(G, source=0, target=7)
print(distance)


# In[369]:



import networkx as nx
import pylab
import numpy as np
#自定义网络
row=np.array([0,1])
col=np.array([1,2])
value=np.array([2,3])
 
print('生成一个空的有向图')
G=nx.DiGraph()
print('为这个网络添加节点...')
for i in range(29):
    G.add_node(i)
print('在网络中添加带权中的边...')
for i in range(np.size(row)):
    G.add_weighted_edges_from([(row[i],col[i],value[i])])
 
print('给网路设置布局...')
pos=nx.shell_layout(G)
print('画出网络图像：')
nx.draw(G,pos,with_labels=True, node_color='white', edge_color='red', node_size=400, alpha=0.5 )
pylab.title('Self_Define Net',fontsize=15)
pylab.show()


# In[458]:


import networkx as nx
import pylab
import numpy as np
#自定义网络
 
print('生成一个空的有向图')
G=nx.Graph()
print('为这个网络添加节点...')
for i in range(29):
    G.add_node(i)
print('在网络中添加带权中的边...')
weighted=[]
for index,row in passing_copy_weight2.iterrows():
    weighted.append(1/row['weight'])
    #G.weight[row['left']][row['right']]=1/row['weight']
    G.add_weighted_edges_from([(row['left'],row['right'],(1/row['weight'])/20)])
#     if(row['left']==20 and row['right']==19):
#         print(row['weight'],'-----------')
    G.edges[row['left'],row['right']]['weight']=(1/row['weight'])/20
 
print('给网路设置布局...')
pos=nx.shell_layout(G)
print('画出网络图像：')
nx.draw(G,pos,with_labels=True, node_color='white', edge_color='red', node_size=40, alpha=0.1 )
pylab.title('DiGraph',fontsize=15)
pylab.show()
def shortest(row):
    return nx.dijkstra_path_length(G, source=row['left'], target=row['right'])
passing_copy_weight2['shortest_path']=passing_copy_weight2.apply(shortest,axis=1)
def onedevideweight(row):
    return 1/row['weight']
passing_copy_weight2['onedevideweight']=passing_copy_weight2.apply(onedevideweight,axis=1)


# In[469]:


import networkx as nx
import pylab
import numpy as np
#自定义网络
 
print('生成一个空的有向图')
G=nx.Graph()
print('为这个网络添加节点...')
for i in range(29):
    G.add_node(i)
print('在网络中添加带权中的边...')
weighted={}
for index,row in passing_copy_weight.iterrows():
    #G.weight[row['left']][row['right']]=1/row['weight']
                #G.add_weighted_edges_from([(row['left'],row['right'],1/row['weight'])])
#     if(row['left']==20 and row['right']==19):
#         print(row['weight'],'-----------')
    #G.add_edge(row['OriginPlayerIDFit'],row['DestinationPlayerIDFit'])
    if (row['OriginPlayerIDFit'],row['DestinationPlayerIDFit']) not in weighted:
        weighted[(row['OriginPlayerIDFit'],row['DestinationPlayerIDFit'])]=1
    else:
        weighted[(row['OriginPlayerIDFit'],row['DestinationPlayerIDFit'])]+=1
   # G.edges[row['OriginPlayerIDFit'],row['DestinationPlayerIDFit']]['weight']+=1
for index,row in passing_copy_weight2.iterrows():
    #G.weight[row['left']][row['right']]=1/row['weight']
    if (row['left'],row['right']) in weighted:
        G.add_weighted_edges_from([(row['left'],row['right'],weighted[(row['left'],row['right'])])])
        G.edges[row['left'],row['right']]['weight']=weighted[(row['left'],row['right'])]
    else:
        G.add_weighted_edges_from([(row['left'],row['right'],0)])
        G.edges[row['left'],row['right']]['weight']=0
#     if(row['left']==20 and row['right']==19):
#         print(row['weight'],'-----------')
    
print('给网路设置布局...')
pos=nx.shell_layout(G)
print('画出网络图像：')
nx.draw(G,pos,with_labels=True, node_color='white', edge_color='red', node_size=40, alpha=0.1 )
pylab.title('DiGraph',fontsize=15)
pylab.show()
def shortest(row):
    return nx.dijkstra_path_length(G, source=row['left'], target=row['right'])
passing_copy_weight2['shortest_path']=passing_copy_weight2.apply(shortest,axis=1)
def onedevideweight(row):
    return 1/row['weight']
passing_copy_weight2['onedevideweight']=passing_copy_weight2.apply(onedevideweight,axis=1)
# for index,row in passing_copy_weight2.iterrows():
#     row['shortest_path']=nx.dijkstra_path_length(G, source=row['left'], target=row['right'])

# '''
# Shortest Path with dijkstra_path
# '''
# print('dijkstra方法寻找最短路径：')
# path=nx.dijkstra_path(G, source=0, target=12)
# print('节点0到12的路径：', path)
# print('dijkstra方法寻找最短距离：')
# distance=nx.dijkstra_path_length(G, source=0, target=12)
# print(distance)


# In[397]:


passing_copy_weight.columns


# In[345]:


from networkx.algorithms.community import kclique as kclique
klist = list(k_clique_communities(G,5))
print(klist)
nx.draw(G,pos =pos, with_labels=False)
nx.draw(G,pos = pos, nodelist = klist[0], node_color = 'b')
nx.draw(G,pos = pos, nodelist = klist[1], node_color = 'y')
plt.show()


# In[470]:


from networkx.algorithms.community import k_clique_communities
from networkx.algorithms.community.label_propagation import asyn_lpa_communities
from networkx.algorithms.community.kernighan_lin import kernighan_lin_bisection
from networkx.algorithms.community.label_propagation import label_propagation_communities
#from networkx.algorithms.community import greedy_modularity_communities

list(asyn_lpa_communities(G,weight='weight'))
#list(greedy_modularity_communities(G,weight='weight'))


# In[482]:


thelist=[{0,1,3,9,16},{2, 4,5,6,7,8,10,11,12,13,14,15,17,18,19,21,22,23,24,25,26,27,28,29},{20}]
for i in range(len(thelist)):
    thelist[i]=[reverse_map_player_id[j] for j in list(thelist[i])]
thelist


# In[116]:


x={0: 0.02040559998827244,1: 0.0,2: 0.014170749726328582,3: 0.016658963483015248,4: 0.014707541888736507,5: 0.009456737684730331,6: 0.008826661685196205,7: 0.006773420017006925,8: 0.0005562418554838753,9: 0.0,10: 0.014170749726328582,11: 0.011862914567508202,12: 0.0009432729767415365,13: 0.007476144386403045,14: 0.006381300251498059,15: 0.007884839385815455,16: 0.01768725579825337,17: 0.0204055999882724418: 0.00013683634373289546,19: 0.0016961230953361382,20: 0.0025843257634386068,21: 0.00012963443090484831,22: 0.00044371196754563894,23: 0.015198977109235767,24: 0.013315973686823477,25: 0.0002988264271225732,26: 0.010834622252270075,27: 0.028: 0.0021043850785230096,29: 0.0039526298443431545}


# In[118]:


for i in list(map_player_id.keys()):
    print(i,x[map_player_id[i]])
    

# In[119]:


one_game=passing_copy_weight[['MatchID','MatchPeriod','EventTime','OriginPlayerIDFit','DestinationPlayerIDFit','weight']]
one_game=one_game[one_game['MatchID']==1]



# In[122]:


def H_event_time_deal(line):
    if line['MatchPeriod']=='2H':
        line['EventTime']+=2712.241
    return line['EventTime']


# In[123]:


one_game['EventTime']=one_game.apply(H_event_time_deal,axis=1)


# ## betweenness_centrality

# In[134]:


interval=350


# In[136]:


import networkx as nx
import pylab
import numpy as np

def centrality_level(passing_copy_weight2,denominator=11):
    #degree,degree_centrality,closeness_centrality,betweenness_centrality,transitivity,clustering
    G=nx.Graph()
    for i in range(30):
        G.add_node(i)
    for index,row in passing_copy_weight2.iterrows():
        G.add_weighted_edges_from([(row['OriginPlayerIDFit'],row['DestinationPlayerIDFit'],1/row['weight'])])
    pos=nx.shell_layout(G)
    return [sum(dict(nx.degree(G)).values())/denominator,            sum(dict(nx.degree_centrality(G)).values())/denominator,            sum(dict(nx.closeness_centrality(G)).values())/denominator,            sum(dict(nx.betweenness_centrality(G)).values())/denominator,            sum(dict(nx.clustering(G)).values())/denominator]


# In[137]:


degree=[]
degree_centrality=[]
clossness_centrality=[]
betweenness_centrality=[]
clustering=[]
betweenness_centrality_2_best=[]
for i in range(16):
    left=i*interval
    right=(i+1)*interval
    print(left,right)
    #between=one_game[(one_game['EventTime']>left) & (one_game['EventTime']<=right)]
    between=one_game[(one_game['EventTime']>left) & (one_game['EventTime']<=right)]
    
    
    between=between[((between['OriginPlayerIDFit']==3) | (between['OriginPlayerIDFit']==17)) | ((between['DestinationPlayerIDFit']==3) | (between['DestinationPlayerIDFit']==17))]
    
    
    between_2_best=between[((between['OriginPlayerIDFit']==16) | (between['OriginPlayerIDFit']==17) | (between['OriginPlayerIDFit']==3)) & ((between['DestinationPlayerIDFit']==16) | (between['DestinationPlayerIDFit']==17) | (between['DestinationPlayerIDFit']==3))]
    degree.append(centrality_level(between)[0])
    betweenness_centrality_2_best.append(centrality_level(between_2_best,2))
    degree_centrality.append(centrality_level(between)[1])
    clossness_centrality.append(centrality_level(between)[2])
    betweenness_centrality.append(centrality_level(between)[3])
#     transitivity.append(centrality_level(between)[4])
    clustering.append(centrality_level(between)[4])


# In[138]:


one_game[((one_game['OriginPlayerIDFit']==0) | (one_game['OriginPlayerIDFit']==2)) & ((one_game['DestinationPlayerIDFit']==0) | (one_game['DestinationPlayerIDFit']==2))]


# In[126]:


degree_centrality_season=[]
clossness_centrality_season=[]
betweenness_centrality_season=[]
clustering_season=[]
for i in range(1,39):
    #game=passing_copy_weight[(passing_copy_weight['MatchID']==i) & (((passing_copy_weight['OriginPlayerIDFit']==3) | (passing_copy_weight['OriginPlayerIDFit']==17)) | ((passing_copy_weight['DestinationPlayerIDFit']==3) | (passing_copy_weight['DestinationPlayerIDFit']==17)))]
    game=passing_copy_weight[(passing_copy_weight['MatchID']==i)]
    
    #between_2_best=between[((between['OriginPlayerIDFit']==16) | (between['OriginPlayerIDFit']==17) | (between['OriginPlayerIDFit']==3)) & ((between['DestinationPlayerIDFit']==16) | (between['DestinationPlayerIDFit']==17) | (between['DestinationPlayerIDFit']==3))]
    #betweenness_centrality_2_best.append(centrality_level(between_2_best,2))
    degree_centrality_season.append(centrality_level(game)[1])
    clossness_centrality_season.append(centrality_level(game)[2])
    betweenness_centrality_season.append(centrality_level(game)[3])
#     transitivity.append(centrality_level(between)[4])
    clustering_season.append(centrality_level(game)[4])


# In[ ]:





# In[127]:


one_game[((one_game['OriginPlayerIDFit']==0) | (one_game['OriginPlayerIDFit']==2)) & ((one_game['DestinationPlayerIDFit']==0) | (one_game['DestinationPlayerIDFit']==2))]


# In[128]:


between=one_game[(one_game['EventTime']>350) & (one_game['EventTime']<=700)]
between=between[((between['OriginPlayerIDFit']==0) | (between['OriginPlayerIDFit']==2)) | ((between['DestinationPlayerIDFit']==0) | (between['DestinationPlayerIDFit']==2))]
between


# In[926]:


import numpy as np 
from matplotlib import pyplot as plt 
import matplotlib
 
# fname 为 你下载的字体库路径，注意 SimHei.ttf 字体的路径
# zhfont1 = matplotlib.font_manager.FontProperties(fname="SimHei.ttf") 
 
x = np.arange(0,5555,350)
 
# fontproperties 设置中文显示，fontsize 设置字体大小
plt.xlabel("time(s)")
plt.ylabel("value")
# plt.plot(x,degree,label="degree") 
plt.plot(x,degree_centrality,label="degree_centrality") 
plt.plot(x,clossness_centrality,label="clossness_centrality") 
plt.plot(x,[25*i for i in betweenness_centrality],label="betweenness_centrality") 
plt.plot(x,[i/4 for i in clustering],label="clustering") 
#plt.legend(["degree","degree_centrality","clossness_centrality",'betweenness_centrality',"clustering"])
plt.legend(["Degree centrality","Clossness centrality",'Betweenness centrality*25',"Clustering/4"])
# plt.show()
plt.savefig("value_of_graph_per_game_2_players.png")


# In[921]:


import numpy as np 
from matplotlib import pyplot as plt 
import matplotlib
 
# fname 为 你下载的字体库路径，注意 SimHei.ttf 字体的路径
# zhfont1 = matplotlib.font_manager.FontProperties(fname="SimHei.ttf") 
 
x = np.arange(0,5555,350)
# degree_centrality=[]
# clossness_centrality=[]
# betweenness_centrality=[]
# clustering=[]
# betweenness_centrality_2_best=[]
# plt.title("value_of_graph_per_game") 
 
# fontproperties 设置中文显示，fontsize 设置字体大小
plt.xlabel("time(s)")
plt.ylabel("value")
# plt.plot(x,degree,label="degree") 
plt.plot(x,degree_centrality,label="degree_centrality") 
plt.plot(x,clossness_centrality,label="clossness_centrality") 
plt.plot(x,[25*i for i in betweenness_centrality],label="betweenness_centrality") 
plt.plot(x,[i/4 for i in clustering],label="clustering") 
#plt.legend(["degree","degree_centrality","clossness_centrality",'betweenness_centrality',"clustering"])
plt.legend(["Degree centrality","Clossness centrality",'Betweenness centrality*25',"Clustering/4"])
# plt.show()
plt.savefig("value_of_graph_per_game.png")


# In[916]:


import numpy as np 
from matplotlib import pyplot as plt 
import matplotlib
 
# zhfont1 = matplotlib.font_manager.FontProperties(fname="SimHei.ttf") 
 
x = np.arange(1,39)
#     degree_centrality_season.append(centrality_level(game)[1])
#     clossness_centrality_season.append(centrality_level(game)[2])
#     betweenness_centrality_season.append(centrality_level(game)[3])
# #     transitivity.append(centrality_level(between)[4])
#     clustering_season.append(centrality_level(game)[4])
# plt.title("value_of_graph_one_season") 

plt.xlabel("game")
plt.ylabel("value")
# plt.plot(x,degree,label="degree") 
plt.plot(x,degree_centrality_season,label="degree_centrality_season") 
plt.plot(x,clossness_centrality_season,label="clossness_centrality_season") 
plt.plot(x,[120*i for i in betweenness_centrality_season],label="betweenness_centrality_season") 
plt.plot(x,[i/4 for i in clustering_season],label="clustering_season") 
#plt.legend(["degree","degree_centrality","clossness_centrality",'betweenness_centrality',"clustering"])
plt.legend(["Degree centrality","Clossness centrality",'Betweenness centrality*120',"Clustering/4"])
# plt.show()
plt.savefig("value_of_graph_one_season.png")


# In[908]:


import numpy as np 
from matplotlib import pyplot as plt 
import matplotlib
 
# fname 为 你下载的字体库路径，注意 SimHei.ttf 字体的路径
# zhfont1 = matplotlib.font_manager.FontProperties(fname="SimHei.ttf") 
 
x = np.arange(1,39)
 
# fontproperties 设置中文显示，fontsize 设置字体大小
plt.xlabel("game")
plt.ylabel("value")
# plt.plot(x,degree,label="degree") 
plt.plot(x,degree_centrality_season,label="degree_centrality_season") 
plt.plot(x,clossness_centrality_season,label="clossness_centrality_season") 
plt.plot(x,[50*i for i in betweenness_centrality_season],label="betweenness_centrality_season") 
plt.plot(x,clustering_season,label="clustering_season") 
#plt.legend(["degree","degree_centrality","clossness_centrality",'betweenness_centrality',"clustering"])
plt.legend(["Degree centrality","Clossness centrality",'Betweenness centrality*50',"Clustering"])
# plt.show()
plt.savefig("value_of_graph_one_season_2_players.png")


# In[ ]:





# ## 所有的球类型

# In[141]:


fullevents=pd.read_csv('fullevents.csv')


# In[144]:


fullevents[fullevents['TeamID']=='Huskies']['OriginPlayerID'].apply(lambda x:x.split('_')[1]).unique()


# In[145]:


h_players=fullevents[fullevents['TeamID']=='Huskies']['OriginPlayerID'].apply(lambda x:x.split('_')[1]).unique()


# In[146]:


o_players=fullevents[fullevents['TeamID']!='Huskies']['OriginPlayerID'].apply(lambda x:x.split('_')[1]).unique()


# In[147]:


init={}
for i in h_players:
    init[i]=[]
init


# In[148]:


init2={}
for i in o_players:
    init2[i]=[]
init2


# In[149]:


init11={}
for i in h_players:
    init11[i]=0
init11
init12={}
for i in o_players:
    init12[i]=0
init12


# In[151]:


def centrality_level_per_person(passing_copy_weight2,denominator=11):
    #degree,degree_centrality,closeness_centrality,betweenness_centrality,transitivity,clustering
    G=nx.Graph()
    for i in range(30):
        G.add_node(i)
    for index,row in passing_copy_weight2.iterrows():
        G.add_weighted_edges_from([(row['OriginPlayerIDFit'],row['DestinationPlayerIDFit'],1/row['weight'])])
    pos=nx.shell_layout(G)
    return dict(nx.betweenness_centrality(G))


# In[152]:


def event_type_score(line):
    event_type=line['EventSubType']
    if event_type=='Ground attacking duel':
        return 4
    elif event_type=='Ground defending duel':
        return 4
    elif event_type=='Ground loose ball duel':
        return 1
    elif event_type=='Air duel':
        return 3
    elif event_type=='Launch':
        return 1 
    elif event_type=='Clearance':
        return 3
    elif event_type=='Smart pass':
        return 5
    elif event_type=='Cross':
        return 2
    elif event_type=='Free kick cross':
        return -2
    elif event_type=='Foul':
        return -7
    elif event_type=='Acceleration':
        return 3
    elif event_type=='Goalkeeper leaving line':
        return 2
    elif event_type=='Out of game foul':
        return -10
    elif event_type=='Save attempt':
        return 4
    elif event_type=='Reflexes':
        return -5
    elif type(event_type)==float:
        return 0
    elif 'foul' in event_type:
        return -8
    else:
        return 0




# In[154]:


betweenness={'H':{'G1':[],'F1':[],'D1':[],'M1':[],'F2':[],'D2':[],'M2':[],'M3':[],'D3':[],'D4':[],'F3':[],'D5':[],'M4':[],'M5':[],'D6':[],'M6':[],'M7':[],'M8':[],'M9':[],'F4':[],'D7':[],'M10':[],'M11':[],'M12':[],'M13':[],'F5':[],'F6':[],'D8':[],'D9':[],'D10':[]},'O':{'D1':[],'D2':[],'M1':[],'G1':[],'F1':[],'F2':[],'D3':[],'M2':[],'F3':[],'M3':[],'D4':[],'F4':[],'F5':[],'M4':[],'D5':[],'M5':[],'M6':[],'G2':[],'M7':[],'D6':[],'F6':[],'D7':[],'D8':[],'M8':[]}} #位置贡献率 球员在自己位置上进行的事件得分
for matchid in range(38):
    #betweenness_this_game={'H':{'G1':0,'F1':0,'D1':0,'M1':0,'F2':0,'D2':0,'M2':0,'M3':0,'D3':0,'D4':0,'F3':0,'D5':0,'M4':0,'M5':0,'D6':0,'M6':0,'M7':0,'M8':0,'M9':0,'F4':0,'D7':0,'M10':0,'M11':0,'M12':0,'M13':0,'F5':0,'F6':0,'D8':0,'D9':0,'D10':0},'O':{'D1':0,'D2':0,'M1':0,'G1':0,'F1':0,'F2':0,'D3':0,'M2':0,'F3':0,'M3':0,'D4':0,'F4':0,'F5':0,'M4':0,'D5':0,'M5':0,'M6':0,'G2':0,'M7':0,'D6':0,'F6':0,'D7':0,'D8':0,'M8':0}}
    this_match_H=passing_copy_weight[passing_copy_weight['MatchID']==(matchid+1)]
    this_match_O=passing_copy_weight_O[passing_copy_weight_O['MatchID']==(matchid+1)]
    y={}
    z={}
    centrality_level_value_H=centrality_level_per_person(this_match_H)
    centrality_level_value_O=centrality_level_per_person(this_match_O)
    for i in list(map_player_id.keys()):
        ii=i.split('_')[1]
        y[ii]=centrality_level_value_H[map_player_id[i]]
    for i in list(map_player_id_O.keys()):
        z[i]=centrality_level_value_O[map_player_id_O[i]]
    for i in list(y.keys()):
        betweenness['H'][i].append(y[i])
    for i in list(z.keys()):
        betweenness['O'][i].append(z[i])
# centrality_level_per_person(this_match)
# for i in list(map_player_id.keys()):
#     print(i,x[map_player_id[i]])
# #betweenness 体现传球 控球率


# In[481]:


reverse_map_player_id={value:key for key,value in map_player_id.items()}


# In[158]:


for i in list(reverse_map_player_id.keys()):
    reverse_map_player_id[i]=reverse_map_player_id[i].split('_')[1]


# In[160]:


def delete0(list_,zero_or_f1):
    a=list(filter(lambda number : number != zero_or_f1, list_))
    if a==[]:
        return [0]
    else:
        return a


# In[161]:


import copy
def get_average(dict,zero_or_f1=0):
    dict_copy=copy.deepcopy(dict)
    dicth=dict_copy['H']
    dicto=dict_copy['O']
    for i in list(dicth.keys()):
        dict_copy['H'][i]=np.mean(delete0(dict_copy['H'][i],zero_or_f1))
    for i in list(dicto.keys()):
        dict_copy['O'][i]=np.mean(delete0(dict_copy['O'][i],zero_or_f1))
    return dict_copy


# In[151]:


#betweenness,event_contribution,possession_charge,total_possession_time,success_guard_percentage,success_shot_percentage


# In[162]:


event_contribution={'H':{'G1':[],'F1':[],'D1':[],'M1':[],'F2':[],'D2':[],'M2':[],'M3':[],'D3':[],'D4':[],'F3':[],'D5':[],'M4':[],'M5':[],'D6':[],'M6':[],'M7':[],'M8':[],'M9':[],'F4':[],'D7':[],'M10':[],'M11':[],'M12':[],'M13':[],'F5':[],'F6':[],'D8':[],'D9':[],'D10':[]},'O':{'D1':[],'D2':[],'M1':[],'G1':[],'F1':[],'F2':[],'D3':[],'M2':[],'F3':[],'M3':[],'D4':[],'F4':[],'F5':[],'M4':[],'D5':[],'M5':[],'M6':[],'G2':[],'M7':[],'D6':[],'F6':[],'D7':[],'D8':[],'M8':[]}} #位置贡献率 球员在自己位置上进行的事件得分
#行为贡献率 球员的行为得分
possession_charge={'H':{'G1':[],'F1':[],'D1':[],'M1':[],'F2':[],'D2':[],'M2':[],'M3':[],'D3':[],'D4':[],'F3':[],'D5':[],'M4':[],'M5':[],'D6':[],'M6':[],'M7':[],'M8':[],'M9':[],'F4':[],'D7':[],'M10':[],'M11':[],'M12':[],'M13':[],'F5':[],'F6':[],'D8':[],'D9':[],'D10':[]},'O':{'D1':[],'D2':[],'M1':[],'G1':[],'F1':[],'F2':[],'D3':[],'M2':[],'F3':[],'M3':[],'D4':[],'F4':[],'F5':[],'M4':[],'D5':[],'M5':[],'M6':[],'G2':[],'M7':[],'D6':[],'F6':[],'D7':[],'D8':[],'M8':[]}} #位置贡献率 球员在自己位置上进行的事件得分
#球权转换率 下一球传给自己人+1 传给对方-2
total_possession_time={'H':{'G1':[],'F1':[],'D1':[],'M1':[],'F2':[],'D2':[],'M2':[],'M3':[],'D3':[],'D4':[],'F3':[],'D5':[],'M4':[],'M5':[],'D6':[],'M6':[],'M7':[],'M8':[],'M9':[],'F4':[],'D7':[],'M10':[],'M11':[],'M12':[],'M13':[],'F5':[],'F6':[],'D8':[],'D9':[],'D10':[]},'O':{'D1':[],'D2':[],'M1':[],'G1':[],'F1':[],'F2':[],'D3':[],'M2':[],'F3':[],'M3':[],'D4':[],'F4':[],'F5':[],'M4':[],'D5':[],'M5':[],'M6':[],'G2':[],'M7':[],'D6':[],'F6':[],'D7':[],'D8':[],'M8':[]}} #位置贡献率 球员在自己位置上进行的事件得分
#控球总时长 按在己方球员操作的时间算


success_guard_percentage={'H':{'G1':[] },'O':{ 'G1':[],'G2':[]}}
success_shot_percentage={'H':{ 'F1':[], 'M1':[],'F2':[], 'M2':[],'M3':[], 'F3':[], 'M4':[],'M5':[], 'M6':[],'M7':[],'M8':[],'M9':[],'F4':[], 'M10':[],'M11':[],'M12':[],'M13':[],'F5':[],'F6':[] },'O':{ 'M1':[],'F1':[],'F2':[], 'M2':[],'F3':[],'M3':[], 'F4':[],'F5':[],'M4':[], 'M5':[],'M6':[],'G2':[],'M7':[], 'F6':[], 'M8':[]}}

for matchid in range(38):
    this_match=fullevents[fullevents['MatchID']==(matchid+1)]
    is_first=1
    #如果是2H加中场结束时间
    half_time=0#上半场该值为0 下半场为上半场结束时间
    is_1H=1
    
    charge_score={'H':{'G1':0,'F1':0,'D1':0,'M1':0,'F2':0,'D2':0,'M2':0,'M3':0,'D3':0,'D4':0,'F3':0,'D5':0,'M4':0,'M5':0,'D6':0,'M6':0,'M7':0,'M8':0,'M9':0,'F4':0,'D7':0,'M10':0,'M11':0,'M12':0,'M13':0,'F5':0,'F6':0,'D8':0,'D9':0,'D10':0},'O':{'D1':0,'D2':0,'M1':0,'G1':0,'F1':0,'F2':0,'D3':0,'M2':0,'F3':0,'M3':0,'D4':0,'F4':0,'F5':0,'M4':0,'D5':0,'M5':0,'M6':0,'G2':0,'M7':0,'D6':0,'F6':0,'D7':0,'D8':0,'M8':0}}
    time={'H':{'G1':0,'F1':0,'D1':0,'M1':0,'F2':0,'D2':0,'M2':0,'M3':0,'D3':0,'D4':0,'F3':0,'D5':0,'M4':0,'M5':0,'D6':0,'M6':0,'M7':0,'M8':0,'M9':0,'F4':0,'D7':0,'M10':0,'M11':0,'M12':0,'M13':0,'F5':0,'F6':0,'D8':0,'D9':0,'D10':0},'O':{'D1':0,'D2':0,'M1':0,'G1':0,'F1':0,'F2':0,'D3':0,'M2':0,'F3':0,'M3':0,'D4':0,'F4':0,'F5':0,'M4':0,'D5':0,'M5':0,'M6':0,'G2':0,'M7':0,'D6':0,'F6':0,'D7':0,'D8':0,'M8':0}}
    event_score={'H':{'G1':0,'F1':0,'D1':0,'M1':0,'F2':0,'D2':0,'M2':0,'M3':0,'D3':0,'D4':0,'F3':0,'D5':0,'M4':0,'M5':0,'D6':0,'M6':0,'M7':0,'M8':0,'M9':0,'F4':0,'D7':0,'M10':0,'M11':0,'M12':0,'M13':0,'F5':0,'F6':0,'D8':0,'D9':0,'D10':0},'O':{'D1':0,'D2':0,'M1':0,'G1':0,'F1':0,'F2':0,'D3':0,'M2':0,'F3':0,'M3':0,'D4':0,'F4':0,'F5':0,'M4':0,'D5':0,'M5':0,'M6':0,'G2':0,'M7':0,'D6':0,'F6':0,'D7':0,'D8':0,'M8':0}}
    guard_count={'H':{'G1':0 },'O':{ 'G1':0,'G2':0}}
    success_guard_count={'H':{'G1':0 },'O':{ 'G1':0,'G2':0}}
    shot_count={'H':{ 'F1':0, 'M1':0,'F2':0, 'M2':0,'M3':0, 'F3':0, 'M4':0,'M5':0, 'M6':0,'M7':0,'M8':0,'M9':0,'F4':0, 'M10':0,'M11':0,'M12':0,'M13':0,'F5':0,'F6':0},'O':{ 'M1':0,'F1':0,'F2':0, 'M2':0,'F3':0,'M3':0, 'F4':0,'F5':0,'M4':0, 'M5':0,'M6':0,'G2':0,'M7':0, 'F6':0, 'M8':0}}
    success_shot_count={'H':{ 'F1':0, 'M1':0,'F2':0, 'M2':0,'M3':0, 'F3':0, 'M4':0,'M5':0, 'M6':0,'M7':0,'M8':0,'M9':0,'F4':0, 'M10':0,'M11':0,'M12':0,'M13':0,'F5':0,'F6':0},'O':{ 'M1':0,'F1':0,'F2':0, 'M2':0,'F3':0,'M3':0, 'F4':0,'F5':0,'M4':0, 'M5':0,'M6':0,'G2':0,'M7':0, 'F6':0, 'M8':0}}
    for index,row in this_match.iterrows():
        if(is_first):
            last_teamid=row['TeamID'][0]#H or O
            last_time=row['EventTime']
            last_player=row['OriginPlayerID'].split('_')[1]
            last_event_type=row['EventType']
            before_turnover_player=[]
            is_first=0
        else:
            if(row['MatchPeriod']=='2H' and is_1H==1):
                half_time=last_time
                is_1H=0
            now_teamid=row['TeamID'][0]
            now_time=row['EventTime']+half_time
            now_player=row['OriginPlayerID'].split('_')[1]
            
            event_score[now_teamid][now_player]+=event_type_score(row)
            if(last_event_type=='Shot'):
                if last_player in shot_count[last_teamid]:
                    shot_count[last_teamid][last_player]+=1
                    if row['EventSubType']=='Reflexes':
                        success_shot_count[last_teamid][last_player]+=1
                if now_player[0]=='G':
                    if row['EventSubType']!='Reflexes':
                        success_guard_count[now_teamid][now_player]+=1
                    guard_count[now_teamid][now_player]+=1
            if (now_teamid==last_teamid):
                time[now_teamid][last_player]+=now_time-last_time
                before_turnover_player.append(last_player)
                for i in before_turnover_player:
                    charge_score[now_teamid][i]+=1
            else:
                charge_score[last_teamid][last_player]-=2
                before_turnover_player=[]
            last_teamid=row['TeamID'][0]
            last_time=row['EventTime']+half_time
            last_player=row['OriginPlayerID'].split('_')[1]
            last_event_type=row['EventType']
    for i in h_players:
        possession_charge['H'][i].append(charge_score['H'][i])
    for i in o_players:
        possession_charge['O'][i].append(charge_score['O'][i])
    for i in h_players:
        total_possession_time['H'][i].append(time['H'][i])
    for i in o_players:
        total_possession_time['O'][i].append(time['O'][i])
    for i in h_players:
        event_contribution['H'][i].append(event_score['H'][i])
    for i in o_players:
        event_contribution['O'][i].append(event_score['O'][i])

    for i in ['G1']:
        success_guard_percentage['H'][i].append(success_guard_count['H'][i]/guard_count['H'][i] if guard_count['H'][i] else -1)
    for i in ['G1','G2']:
        success_guard_percentage['O'][i].append(success_guard_count['O'][i]/guard_count['O'][i] if guard_count['O'][i] else -1)
    for i in list(shot_count['H'].keys()):
        success_shot_percentage['H'][i].append(success_shot_count['H'][i]/shot_count['H'][i] if shot_count['H'][i] else -1)
    for i in list(shot_count['O'].keys()):
        success_shot_percentage['O'][i].append(success_shot_count['O'][i]/shot_count['O'][i] if shot_count['O'][i] else -1)


# In[222]:


event_contribution


# In[154]:


import numpy as np 
from matplotlib import pyplot as plt 
import matplotlib
 
# fname 为 你下载的字体库路径，注意 SimHei.ttf 字体的路径
# zhfont1 = matplotlib.font_manager.FontProperties(fname="SimHei.ttf") 
 
x = np.arange(0,11)
 
# fontproperties 设置中文显示，fontsize 设置字体大小
plt.xlabel("playerID")
plt.ylabel("value")
# plt.plot(x,degree,label="degree") 
plt.plot(x,sorted([10000*v for k, v in get_average(betweenness)['H'].items()],reverse=True)[:11],label="betweenness*10000") 
plt.plot(x,sorted([v for k, v in get_average(event_contribution)['H'].items()],reverse=True)[:11],label="event_contribution") 
plt.plot(x,sorted([v for k, v in get_average(possession_charge)['H'].items()],reverse=True)[:11],label="possession_charge") 
plt.plot(x,sorted([v/1.5 for k, v in get_average(total_possession_time)['H'].items()],reverse=True)[:11],label="total_possession_time/1.5") 
# plt.plot(x,[v for k, v in get_average(betweenness)['H'].items()],label="betweenness") 
#plt.legend(["degree","degree_centrality","clossness_centrality",'betweenness_centrality',"clustering"])
plt.legend(["Betweenness*10000",'Event contribution','Possession charge','Total possession time/1.5'])
# plt.show()
plt.savefig("value_of_person_per_season_Huskies_________.png")


# In[1199]:


import numpy as np 
from matplotlib import pyplot as plt 
import matplotlib
 
x = np.arange(0,30)

plt.xlabel("playerID")
plt.ylabel("value")
# plt.plot(x,degree,label="degree") 
plt.plot(x,[10000*v for k, v in get_average(betweenness)['H'].items()],label="betweenness*10000") 
plt.plot(x,[v for k, v in get_average(event_contribution)['H'].items()],label="event_contribution") 
plt.plot(x,[v for k, v in get_average(possession_charge)['H'].items()],label="possession_charge") 
plt.plot(x,[v/1.5 for k, v in get_average(total_possession_time)['H'].items()],label="total_possession_time/1.5") 
# plt.plot(x,[v for k, v in get_average(betweenness)['H'].items()],label="betweenness") 
#plt.legend(["degree","degree_centrality","clossness_centrality",'betweenness_centrality',"clustering"])
plt.legend(["Betweenness*10000",'Event contribution','Possession charge','Total possession time/1.5'])
# plt.show()
plt.savefig("value_of_player_per_season_Huskies_30player.png")


# In[1198]:


import numpy as np 
from matplotlib import pyplot as plt 
import matplotlib

 
x = np.arange(0,24)

plt.xlabel("playerID")
plt.ylabel("value")
# plt.plot(x,degree,label="degree") 
plt.plot(x,[10000*v for k, v in get_average(betweenness)['O'].items()],label="betweenness*10000") 
plt.plot(x,[v for k, v in get_average(event_contribution)['O'].items()],label="event_contribution") 
plt.plot(x,[v for k, v in get_average(possession_charge)['O'].items()],label="possession_charge") 
plt.plot(x,[v/1.5 for k, v in get_average(total_possession_time)['O'].items()],label="total_possession_time/1.5") 
# plt.plot(x,[v for k, v in get_average(betweenness)['H'].items()],label="betweenness") 
#plt.legend(["degree","degree_centrality","clossness_centrality",'betweenness_centrality',"clustering"])
plt.legend(["Betweenness*10000",'Event contribution','Possession charge','Total possession time/1.5'])
# plt.show()
plt.savefig("value_of_player_per_season_Opponents_24player.png")



# In[895]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family']='Times New Roman'
matplotlib.rcParams['font.sans-serif']=['Times New Roman']
labels=np.array(['Betweenness*10000','Event contribution','Possession charge','Total possession time/2','Success shot percentage'])
nAttr=5
Python=np.array([get_average(betweenness)['H']['M1']*10000/sorted([10000*v for k, v in get_average(betweenness)['H'].items()],reverse=True)[:1]*100,                get_average(event_contribution)['H']['M1']/sorted([v for k, v in get_average(event_contribution)['H'].items()],reverse=True)[:1]*100,                get_average(possession_charge)['H']['M1']/sorted([v for k, v in get_average(possession_charge)['H'].items()],reverse=True)[:1]*100,                get_average(total_possession_time)['H']['M1']/2/sorted([v/2 for k, v in get_average(total_possession_time)['H'].items()],reverse=True)[:1]*100,                get_average(success_shot_percentage,-1)['H']['M1']/sorted([v for k, v in get_average(success_shot_percentage,-1)['H'].items()],reverse=True)[:1]*100])
Python2=np.array([get_average(betweenness)['O']['M1']*10000/sorted([10000*v for k, v in get_average(betweenness)['O'].items()],reverse=True)[:1]*100,                get_average(event_contribution)['O']['M1']/sorted([v for k, v in get_average(event_contribution)['O'].items()],reverse=True)[:1]*100,                get_average(possession_charge)['O']['M1']/sorted([v for k, v in get_average(possession_charge)['O'].items()],reverse=True)[:1]*100,                get_average(total_possession_time)['O']['M1']/2/sorted([v/2 for k, v in get_average(total_possession_time)['O'].items()],reverse=True)[:1]*100,                get_average(success_shot_percentage,-1)['O']['M1']/sorted([v for k, v in get_average(success_shot_percentage,-1)['O'].items()],reverse=True)[:1]*100])

angles=np.linspace(0,2*np.pi,nAttr,endpoint=False)
Python=np.concatenate((Python,[Python[0]]))
Python2=np.concatenate((Python2,[Python2[0]]))
angles=np.concatenate((angles,[angles[0]]))
fig=plt.figure(facecolor="white")
plt.subplot(111,polar=True)
plt.plot(angles,Python,'bo-',color='g',linewidth=2)
plt.fill(angles,Python,facecolor='g',alpha=0.2)
plt.plot(angles,Python2,'bo-',color='r',linewidth=2)
plt.fill(angles,Python2,facecolor='r',alpha=0.2)
plt.thetagrids(angles*180/np.pi,labels)
plt.figtext(0.52,0.95,'Huskies_M1(green) VS Opponents_M1(red)',size=12,ha='center')
plt.grid(True)
plt.savefig('dota_radar.JPG')
plt.show()


# In[251]:


SHUFFLE=delete0([i if 'G' in i else 0 for i in [k for k,v in betweenness['H'].items()]],0)+delete0([i if 'D' in i else 0 for i in [k for k,v in betweenness['H'].items()]],0)+delete0([i if 'F' in i else 0 for i in [k for k,v in betweenness['H'].items()]],0)+delete0([i if 'M' in i else 0 for i in [k for k,v in betweenness['H'].items()]],0)


# In[254]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family']='Times New Roman'
matplotlib.rcParams['font.sans-serif']=['Times New Roman']
bt=[]
ec=[]
pc=[]
tpt=[]
for i in SHUFFLE:
    bt.append(get_average(betweenness)['H'][i]*10000*Q[0].real)
    ec.append(get_average(event_contribution)['H'][i]*Q[1].real)
    pc.append(get_average(possession_charge)['H'][i]*Q[2].real)
    tpt.append(get_average(total_possession_time)['H'][i]/2*Q[3].real)
x=range(len([k for k,v in betweenness['H'].items()]))
plt.figure(figsize=(15,10))
# 堆积柱状图
plt.bar(x, bt, color='r', label='Betweenness*10000*Q[0]')
plt.bar(x, ec, bottom=np.array(bt), color='#000000', label='Event Contribution*Q[1]')
plt.bar(x, pc, bottom=np.array(bt)+np.array(ec), color='b', label='Poesession Charge*Q[2]')
plt.bar(x, tpt, bottom=np.array(bt)+np.array(ec)+np.array(pc), color='c', label='Total Possession Time/2*Q[3]')
# # 显示范围
# plt.xlim(-2, 22)
# plt.ylim(0, 280)

# 添加图例
plt.legend(loc='upper right')

plt.xticks([index + 0.2 for index in x], SHUFFLE)
plt.xlabel("Player")
plt.ylabel("Score")
plt.grid(axis='y', color='gray', linestyle=':', linewidth=2)
plt.savefig('player_score_bar.png')
plt.show()


# In[552]:


#bt,ec,pc,tpt
x0=np.array([i/Q[0].real for i in list(bt)])
x1=np.array([i/Q[1].real for i in list(ec)])
x2=np.array([i/Q[2].real for i in list(pc)])
x3=np.array([i/Q[3].real for i in list(tpt)])
def cal(list_):
    return np.mean(list_[0]*x0+list_[1]*x1+list_[2]*x2+list_[3]*x3)
table.iloc[0,0:4]=[Q[0].real,Q[1].real,Q[2].real,Q[3].real]
table.iloc[0,4]=0
table.iloc[0,5]=cal(table.iloc[0,0:4])
table.iloc[0,6]=0

table.iloc[1,0:4]=[Q[0].real*(1+0.15),Q[1].real,Q[2].real,Q[3].real]
table.iloc[1,4]=0.15
table.iloc[1,5]=cal(table.iloc[1,0:4])
table.iloc[1,6]=table.iloc[1,5]/table.iloc[0,5]-1
table.iloc[2,0:4]=[Q[0].real*(1-0.15),Q[1].real,Q[2].real,Q[3].real]
table.iloc[2,4]=-0.15
table.iloc[2,5]=cal(table.iloc[2,0:4])
table.iloc[2,6]=table.iloc[2,5]/table.iloc[0,5]-1

table.iloc[3,0:4]=[Q[0].real,Q[1].real*(1+0.15),Q[2].real,Q[3].real]
table.iloc[3,4]=0.15
table.iloc[3,5]=cal(table.iloc[3,0:4])
table.iloc[3,6]=table.iloc[3,5]/table.iloc[0,5]-1
table.iloc[4,0:4]=[Q[0].real,Q[1].real*(1-0.15),Q[2].real,Q[3].real]
table.iloc[4,4]=-0.15
table.iloc[4,5]=cal(table.iloc[4,0:4])
table.iloc[4,6]=table.iloc[4,5]/table.iloc[0,5]-1

table.iloc[5,0:4]=[Q[0].real,Q[1].real,Q[2].real*(1+0.15),Q[3].real]
table.iloc[5,4]=0.15
table.iloc[5,5]=cal(table.iloc[5,0:4])
table.iloc[5,6]=table.iloc[5,5]/table.iloc[0,5]-1
table.iloc[6,0:4]=[Q[0].real,Q[1].real,Q[2].real*(1-0.15),Q[3].real]
table.iloc[6,4]=-0.15
table.iloc[6,5]=cal(table.iloc[6,0:4])
table.iloc[6,6]=table.iloc[6,5]/table.iloc[0,5]-1

table.iloc[7,0:4]=[Q[0].real,Q[1].real,Q[2].real,Q[3].real*(1+0.15)]
table.iloc[7,4]=0.15
table.iloc[7,5]=cal(table.iloc[7,0:4])
table.iloc[7,6]=table.iloc[7,5]/table.iloc[0,5]-1
table.iloc[8,0:4]=[Q[0].real,Q[1].real,Q[2].real,Q[3].real*(1-0.15)]
table.iloc[8,4]=-0.15
table.iloc[8,5]=cal(table.iloc[8,0:4])
table.iloc[8,6]=table.iloc[8,5]/table.iloc[0,5]-1


# In[547]:


#table.iloc[[0,2],0:3]=[[0,0,0],[0,0,1]]
table.iloc[0,0:7]


# In[524]:


table['betweenness'][0]


# In[551]:


table=pd.DataFrame(np.zeros([9,7])).rename(columns={0:'betweenness',1:'event_contribution',2:'possession_charge',3:'total_possession_time',4:'changing_proportion',5:'average_score',6:'deviation_of_score'})


# In[554]:


table.to_csv('mingganxing.csv')


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family']='Times New Roman'
matplotlib.rcParams['font.sans-serif']=['Times New Roman']
labels=np.array(['Betweenness*10000','Event contribution','Possession charge','Total possession time/2','Success shot percentage'])
nAttr=5
Python=np.array([get_average(betweenness)['H']['M1']*10000/sorted([10000*v for k, v in get_average(betweenness)['H'].items()],reverse=True)[:1]*100,                get_average(event_contribution)['H']['M1']/sorted([v for k, v in get_average(event_contribution)['H'].items()],reverse=True)[:1]*100,                get_average(possession_charge)['H']['M1']/sorted([v for k, v in get_average(possession_charge)['H'].items()],reverse=True)[:1]*100,                get_average(total_possession_time)['H']['M1']/2/sorted([v/2 for k, v in get_average(total_possession_time)['H'].items()],reverse=True)[:1]*100,                get_average(success_shot_percentage,-1)['H']['M1']/sorted([v for k, v in get_average(success_shot_percentage,-1)['H'].items()],reverse=True)[:1]*100])
Python2=np.array([get_average(betweenness)['O']['M1']*10000/sorted([10000*v for k, v in get_average(betweenness)['O'].items()],reverse=True)[:1]*100,                get_average(event_contribution)['O']['M1']/sorted([v for k, v in get_average(event_contribution)['O'].items()],reverse=True)[:1]*100,                get_average(possession_charge)['O']['M1']/sorted([v for k, v in get_average(possession_charge)['O'].items()],reverse=True)[:1]*100,                get_average(total_possession_time)['O']['M1']/2/sorted([v/2 for k, v in get_average(total_possession_time)['O'].items()],reverse=True)[:1]*100,                get_average(success_shot_percentage,-1)['O']['M1']/sorted([v for k, v in get_average(success_shot_percentage,-1)['O'].items()],reverse=True)[:1]*100])

angles=np.linspace(0,2*np.pi,nAttr,endpoint=False)
Python=np.concatenate((Python,[Python[0]]))
Python2=np.concatenate((Python2,[Python2[0]]))
angles=np.concatenate((angles,[angles[0]]))
fig=plt.figure(facecolor="white")
plt.subplot(111,polar=True)
plt.plot(angles,Python,'bo-',color='g',linewidth=2)
plt.fill(angles,Python,facecolor='g',alpha=0.2)
plt.plot(angles,Python2,'bo-',color='r',linewidth=2)
plt.fill(angles,Python2,facecolor='r',alpha=0.2)
plt.thetagrids(angles*180/np.pi,labels)
plt.figtext(0.52,0.95,'Huskies_M1(green) VS Opponents_M1(red)',size=12,ha='center')
plt.grid(True)
plt.savefig('dota_radar.JPG')
plt.show()


# In[229]:


import matplotlib.pyplot as plt
import numpy as np

# 生成数据
x = np.linspace(0, 20, 20)
y1 = np.random.randint(50, 100, 20)
y2 = np.random.randint(50, 100, 20)
y3 = np.random.randint(50, 100, 20)

# 堆积柱状图
plt.bar(x, y1, color='r', label='语文')
plt.bar(x, y2, bottom=y1, color='g', label='数学')
plt.bar(x, y3, bottom=y1+y2, color='c', label='英语')

# 显示范围
plt.xlim(-2, 22)
plt.ylim(0, 280)

# 添加图例
plt.legend(loc='upper right')
plt.grid(axis='y', color='gray', linestyle=':', linewidth=2)

plt.show()


# In[804]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family']='SimHei'
matplotlib.rcParams['font.sans-serif']=['SimHei']
labels=np.array(['综合','第一周','第二周','第三周','第四周'])
nAttr=5
Python=np.array([1,85,90,95,70])
angles=np.linspace(0,2*np.pi,nAttr,endpoint=False)
Python=np.concatenate((Python,[Python[0]]))
angles=np.concatenate((angles,[angles[0]]))
fig=plt.figure(facecolor="white")
plt.subplot(111,polar=True)
plt.plot(angles,Python,'bo-',color='g',linewidth=2)
plt.fill(angles,Python,facecolor='g',alpha=0.2)
plt.thetagrids(angles*180/np.pi,labels)
# plt.figtext(0.52,0.95,'python成绩分析图',ha='center')
# plt.grid(True)
plt.savefig('dota_radar.JPG')
plt.show()


# In[907]:


import numpy as np 
from matplotlib import pyplot as plt 
import matplotlib
 
x = np.arange(0,11)
 
plt.xlabel("playerID")
plt.ylabel("value")
# plt.plot(x,degree,label="degree") 
plt.plot(x,sorted([10000*v for k, v in get_average(betweenness)['O'].items()],reverse=True)[:11],label="betweenness*10000") 
plt.plot(x,sorted([v for k, v in get_average(event_contribution)['O'].items()],reverse=True)[:11],label="event_contribution") 
plt.plot(x,sorted([v/2 for k, v in get_average(possession_charge)['O'].items()],reverse=True)[:11],label="possession_charge/2") 
plt.plot(x,sorted([v/2 for k, v in get_average(total_possession_time)['O'].items()],reverse=True)[:11],label="total_possession_time/2") 
# plt.plot(x,[v for k, v in get_average(betweenness)['H'].items()],label="betweenness") 
#plt.legend(["degree","degree_centrality","clossness_centrality",'betweenness_centrality',"clustering"])
plt.legend(["Betweenness*10000",'Event contribution','Possession charge/2','Total possession time/2'])
# plt.show()
plt.savefig("value_of_person_per_season_Opponents.png")


# ## DBSCAN

# In[166]:


import sklearn.cluster as skc  # 密度聚类
dbscan=skc.DBSCAN(eps=1, min_samples=2).fit(StandardScaler().fit_transform( np.array([[v for k, v in get_average(betweenness)['H'].items()], [v for k, v in get_average(event_contribution)['H'].items()], [v for k, v in get_average(possession_charge)['H'].items()], [v for k, v in get_average(total_possession_time)['H'].items()]]).reshape(30,4)))
dbscan.labels_


# In[ ]:


np.array(StandardScaler().fit_transform([[v*10000 for k, v in get_average(betweenness)['H'].items()], [v for k, v in get_average(event_contribution)['H'].items()], [v for k, v in get_average(possession_charge)['H'].items()], [v/1.5 for k, v in get_average(total_possession_time)['H'].items()]])).reshape(30,4)


# In[165]:


from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)

# X = np.array(sorted(np.array([[v for k, v in get_average(betweenness)['H'].items()],\
# #  [v for k, v in get_average(event_contribution)['H'].items()],\
# #  [v for k, v in get_average(possession_charge)['H'].items()],\
#  [v for k, v in get_average(total_possession_time)['H'].items()]]).reshape(30,2),key=lambda x:(pow(x[0],2)+pow(x[1],2)),reverse=True)[:11])

X = np.array([[v for k, v in get_average(betweenness)['H'].items()], [v for k, v in get_average(event_contribution)['H'].items()], [v for k, v in get_average(possession_charge)['H'].items()], [v for k, v in get_average(total_possession_time)['H'].items()]]).reshape(30,4)



# X=np.array([float(i) for i in sorted(np.array(PlayerScore).reshape(-1,1),key=lambda x:x[0],reverse=True)[:11]]).reshape(-1,1)


# In[174]:


db = DBSCAN(eps=40, min_samples=3).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1] ,'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1],'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()


# In[1117]:


np.array([float(i) for i in sorted(np.array(PlayerScore).reshape(-1,1),key=lambda x:x[0],reverse=True)[:11]]).reshape(-1,1)


# In[1087]:


np.array([[v for k, v in get_average(betweenness)['H'].items()],#  [v for k, v in get_average(event_contribution)['H'].items()],\
#  [v for k, v in get_average(possession_charge)['H'].items()],\
 [v for k, v in get_average(total_possession_time)['H'].items()]]).reshape(30,2)


# In[1090]:


sorted(np.array([[v for k, v in get_average(betweenness)['H'].items()],#  [v for k, v in get_average(event_contribution)['H'].items()],\
#  [v for k, v in get_average(possession_charge)['H'].items()],\
 [v for k, v in get_average(total_possession_time)['H'].items()]]).reshape(30,2),key=lambda x:(pow(x[0],2)+pow(x[1],2)),reverse=True)[:15]


# In[1008]:


labels = db.labels_


# In[167]:


import numpy as np

A = np.array([[1, 5, 3,7],
   [1/5,1,3/5,3/7],
   [1/3,5/3,1,7/5],
    [1/7,7/3,5/7,1]])
m=len(A)                                    #获取数据个数
n=len(A[0])
RI=[0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51]
R= np.linalg.matrix_rank(A)                #求判断矩阵的秩
V,D=np.linalg.eig(A)                       #求判断矩阵的特征值和特征向量，V特征值，D特征向量；
list1 = list(V)
B= np.max(list1)                           #最大特征值
index = list1.index(B)
C = D[:, index]                            #对应特征向量
CI=(B-n)/(n-1)                             #计算一致性检验指标CI
CR=CI/RI[n]
if CR<0.10:
    print("CI=", CI)
    print("CR=", CR)
    print('对比矩阵A通过一致性检验，各向量权重向量Q为：')
    sum=np.sum(C)

    Q=C/sum                               #特征向量标准化
    print(Q)                              #  输出权重向量
else:
    print("对比矩阵A未通过一致性检验，需对对比矩阵A重新构造")


# In[1046]:


A = np.array([[1,6,0.5],
              [0.166,1,0.25],
              [2,4,1]])
SUMR = []
for i in range(0,3):
    tempsum=0
    for j in range(0,3):
        tempsum+=A[j][i]
    SUMR.append(tempsum)
A=np.row_stack((A,SUMR))
A


# In[169]:


Final_Score={'H':{'G1':[],'F1':[],'D1':[],'M1':[],'F2':[],'D2':[],'M2':[],'M3':[],'D3':[],'D4':[],'F3':[],'D5':[],'M4':[],'M5':[],'D6':[],'M6':[],'M7':[],'M8':[],'M9':[],'F4':[],'D7':[],'M10':[],'M11':[],'M12':[],'M13':[],'F5':[],'F6':[],'D8':[],'D9':[],'D10':[]},'O':{'D1':[],'D2':[],'M1':[],'G1':[],'F1':[],'F2':[],'D3':[],'M2':[],'F3':[],'M3':[],'D4':[],'F4':[],'F5':[],'M4':[],'D5':[],'M5':[],'M6':[],'G2':[],'M7':[],'D6':[],'F6':[],'D7':[],'D8':[],'M8':[]}} #位置贡献率 球员在自己位置上进行的事件得分
for i in [k for k, v in betweenness['H'].items()]:
    a=0.9*(Q[0].real*10000*np.array(betweenness['H'][i])+                            Q[1].real*np.array(event_contribution['H'][i])+                            Q[2].real*np.array(possession_charge['H'][i])+                            Q[3].real*1/1.5*np.array(total_possession_time['H'][i]))
    if i in success_guard_percentage['H']:
        a+=0.05*100*(Q[0].real*np.array([j if j!=-1 else 0 for j in success_guard_percentage['H'][i]]))
    if i in success_shot_percentage['H']:
        a+=0.05*100*(Q[0].real*np.array([j if j!=-1 else 0 for j in success_shot_percentage['H'][i]]))
    Final_Score['H'][i]=list(a)
for i in [k for k, v in betweenness['O'].items()]:
    if betweenness['O'][i]==[]:
        a=0.9*(                            Q[1].real*np.array(event_contribution['O'][i])+                            Q[2].real*np.array(possession_charge['O'][i])+                            Q[3].real*1/1.5*np.array(total_possession_time['O'][i]))
    else:
        a=0.9*(Q[0].real*10000*np.array(betweenness['O'][i])+                            Q[1].real*np.array(event_contribution['O'][i])+                            Q[2].real*np.array(possession_charge['O'][i])+                            Q[3].real*1/1.5*np.array(total_possession_time['O'][i]))
    if i in success_guard_percentage['O']:
        a+=0.05*100*(Q[0].real*np.array([j if j!=-1 else 0 for j in success_guard_percentage['O'][i]]))
    if i in success_shot_percentage['O']:
        a+=0.05*100*(Q[0].real*np.array([j if j!=-1 else 0 for j in success_shot_percentage['O'][i]]))
    Final_Score['O'][i]=list(a)
Final_Score


# In[181]:


AvgFinalScore={'H':{'G1':0,'F1':0,'D1':0,'M1':0,'F2':0,'D2':0,'M2':0,'M3':0,'D3':0,'D4':0,'F3':0,'D5':0,'M4':0,'M5':0,'D6':0,'M6':0,'M7':0,'M8':0,'M9':0,'F4':0,'D7':0,'M10':0,'M11':0,'M12':0,'M13':0,'F5':0,'F6':0,'D8':0,'D9':0,'D10':0},'O':{'D1':0,'D2':0,'M1':0,'G1':0,'F1':0,'F2':0,'D3':0,'M2':0,'F3':0,'M3':0,'D4':0,'F4':0,'F5':0,'M4':0,'D5':0,'M5':0,'M6':0,'G2':0,'M7':0,'D6':0,'F6':0,'D7':0,'D8':0,'M8':0}}
for i in [k for k, v in Final_Score['H'].items()]:
    AvgFinalScore['H'][i]=np.mean(delete0(Final_Score['H'][i],0))
for i in [k for k, v in Final_Score['O'].items()]:
    AvgFinalScore['O'][i]=np.mean(delete0(Final_Score['O'][i],0))
AvgFinalScore


# In[1185]:


AvgFinalScore['H']=sorted(AvgFinalScore['H'].items(),key=lambda x:x[1],reverse=True)
AvgFinalScore['O']=sorted(AvgFinalScore['O'].items(),key=lambda x:x[1],reverse=True)


# # 一个队一个分


# In[176]:



def pos(line):#G_count FM_count
    return line['OriginPlayerID'].split('_')[1]
def count_pos(df):
    g=0
    fm=0
    xx=df.apply(pos,axis=1).unique()
    for i in xx:
        if 'G' in i:
            g+=1
        if 'F' in i or 'M' in i:
            fm+=1
    return(g,fm)




# In[202]:


fullevents[fullevents['TeamID']=='Huskies'].ix[[5]]


# In[ ]:


#一个客队两场比赛 #betweenness_per_team



# In[173]:


def centrality_level_pc(passing_copy_weight2,denominator=11):
    #degree,degree_centrality,closeness_centrality,betweenness_centrality,transitivity,clustering
    G=nx.Graph()
    for i in range(30):
        G.add_node(i)
    for index,row in passing_copy_weight2.iterrows():
        G.add_weighted_edges_from([(row['OriginPlayerIDFit'],row['DestinationPlayerIDFit'],1/row['weight'])])
    pos=nx.shell_layout(G)
    return np.mean(delete0([v for k,v in dict(nx.betweenness_centrality(G)).items()],0))


# In[177]:


fullevents.columns


# In[183]:


fullevents['DestinationPlayerID'].isnull()


# In[ ]:





# In[170]:


fullevents_D=fullevents[fullevents['DestinationPlayerID'].notnull()]


# In[171]:


fullevents_D['OriginPlayerIDFit']=preprocessing.LabelEncoder().fit_transform(fullevents_D['OriginPlayerID'])
fullevents_D['DestinationPlayerIDFit']=preprocessing.LabelEncoder().fit_transform(fullevents_D['DestinationPlayerID'])


# In[174]:


betweenness_per_team={}
all_=pd.concat([passing_copy_weight,passing_copy_weight_O],axis=0)
for teamid in all_['TeamID'].unique():
    game_per_team=all_[all_['TeamID']==teamid]
    player_count=len(game_per_team['OriginPlayerID'].unique())
    aa=0
    tot=0
    for i in game_per_team['MatchID'].unique():
        gpt=game_per_team[game_per_team['MatchID']==i]
        tot+=centrality_level_pc(gpt)
        aa+=1
    betweenness_per_team[teamid]=tot/aa


# In[206]:


betweenness_per_team


# In[178]:


import warnings 
warnings.filterwarnings('ignore')

event_contribution_per_team={}
possession_charge_per_team={}
total_possession_time_per_team={}
success_guard_percentage_per_team={}
success_shot_percentage_per_team={}
for teamid in fullevents['TeamID'].unique():
    game_per_team=fullevents[fullevents['TeamID']==teamid]
    player_count=len(game_per_team['OriginPlayerID'].unique())
    
    G_count=count_pos(game_per_team)[0]
    FM_count=count_pos(game_per_team)[1]
    is_first=1
    #如果是2H加中场结束时间
    half_time=0#如果有上下半场交换 时长加上上一次half_time      
    half_time2=0
    charge_score=0#+1-2 球权转换率
    time=0 #控球时间
    event_score=0 #事件得分
    guard_count=0 #防守次数
    success_guard_count=0#防守成功次数
    shot_count=0#射门次数
    success_shot_count=0#射门成功次数
        
    for index,row in game_per_team.iterrows():
        if(is_first):
            is_first=0
            before_turnover_player=0
            last_period=row['MatchPeriod']
        else:
            last_teamid=fullevents.ix[[index-1]]['TeamID'].iloc[0][0]
            last_period_to_cal=fullevents.ix[[index-1]]['MatchPeriod'].iloc[0]
            last_event_type=fullevents.ix[[index-1]]['EventType'].iloc[0]
            last_time_to_cal=fullevents.ix[[index-1]]['EventTime'].iloc[0]+half_time
            now_teamid=fullevents.ix[[index]]['TeamID'].iloc[0][0]
            
            now_player=row['OriginPlayerID'].split('_')[1]
            now_period=row['MatchPeriod']
            if(last_period!=now_period):
                half_time=last_time
            if(last_period_to_cal==now_period):
                last_time_to_cal=fullevents.ix[[index-1]]['EventTime'].iloc[0]+half_time
            now_time=row['EventTime']+half_time
            #print('---',now_time,last_time_to_cal,'---')
            event_score+=event_type_score(row)
            if(last_event_type=='Shot' and last_teamid!=teamid[0]):
                print(last_teamid,teamid,row['EventType'],now_teamid,row['EventSubType'],guard_count,success_guard_count)
                if row['EventType']=='Save Attempt' and now_teamid==teamid[0]:
                    guard_count+=1
                    if row['EventSubType']!='Reflexes':
                        success_guard_count+=1
            if(last_event_type=='Shot' and last_teamid==teamid[0]):
                if row['EventType']=='Save Attempt' and now_teamid!=teamid[0]:
                    shot_count+=1
                    if row['EventSubType']=='Reflexes':
                        success_shot_count+=1
            if (now_teamid==last_teamid):
                #if(now_time<last_time_to_cal or now_time-last_time_to_cal>1000):
                #    print(now_time,last_time_to_cal,last_time,half_time)
                time+=now_time-last_time_to_cal
                before_turnover_player+=1
                charge_score+=before_turnover_player
            else:
                charge_score-=2
                before_turnover_player=0
            last_time=row['EventTime']+half_time
            last_period=row['MatchPeriod']
    event_contribution_per_team[teamid]=event_score/player_count/2
    possession_charge_per_team[teamid]=charge_score/player_count/2
    total_possession_time_per_team[teamid]=time/player_count/2
    success_guard_percentage_per_team[teamid]=success_guard_count/guard_count/G_count/2 if guard_count!=0 else 0
    success_shot_percentage_per_team[teamid]=success_shot_count/shot_count/FM_count/2 if shot_count!=0 else 0


# In[180]:


event_contribution_per_team['Huskies']/=19/2
possession_charge_per_team['Huskies']/=19/2
total_possession_time_per_team['Huskies']/=19/2
success_guard_percentage_per_team['Huskies']/=19/2
success_shot_percentage_per_team['Huskies']/=19/2


# In[210]:


team_score={}
for i in [k for k, v in total_possession_time_per_team.items()]:
    team_score[i]=(Q[0].real*10000*betweenness_per_team[i]+                            Q[1].real*event_contribution_per_team[i]+                            Q[2].real*possession_charge_per_team[i]+                            Q[3].real*1/1.5*total_possession_time_per_team[i])


#betweenness


# In[185]:


len_=0
H_score_per_game=np.zeros((38,))
zero=np.zeros((38,))
for i in ([np.array(v) for k,v in betweenness['H'].items()]):
    zero+=i
    len_+=1
H_score_per_game+=zero/len_*10000*Q[0].real
len_=0
zero=np.zeros((38,))
for i in ([np.array(v) for k,v in event_contribution['H'].items()]):
    zero+=i
    len_+=1
H_score_per_game+=zero/len_*Q[1].real
len_=0
zero=np.zeros((38,))
for i in ([np.array(v) for k,v in possession_charge['H'].items()]):
    zero+=i
    len_+=1
H_score_per_game+=zero/len_*Q[2].real
len_=0
zero=np.zeros((38,))
for i in ([np.array(v) for k,v in total_possession_time['H'].items()]):
    zero+=i
    len_+=1
H_score_per_game+=zero/len_/1.5*Q[3].real
H_score_per_game=H_score_per_game/22.837916545013048*50.31779008693441
H_score_per_game


# In[403]:


len_=0
O_score_per_game=np.zeros((38,))
zero=np.zeros((38,))
for i in ([np.array(v) for k,v in betweenness['O'].items()]):
    if i!=[]:
        zero+=i
        len_+=1
O_score_per_game+=zero/len_*10000*Q[0].real
len_=0
zero=np.zeros((38,))
for i in ([np.array(v) for k,v in event_contribution['O'].items()]):
    zero+=i
    len_+=1
O_score_per_game+=zero/len_*Q[1].real
len_=0
zero=np.zeros((38,))
for i in ([np.array(v) for k,v in possession_charge['O'].items()]):
    zero+=i
    len_+=1
O_score_per_game+=zero/len_*Q[2].real
len_=0
zero=np.zeros((38,))
for i in ([np.array(v) for k,v in total_possession_time['O'].items()]):
    zero+=i
    len_+=1
O_score_per_game+=zero/len_/1.5*Q[3].real
O_score_per_game=O_score_per_game/22.837916545013048*50.31779008693441
O_score_per_game


# In[186]:


betweenness['O']['D1']


# In[415]:


import numpy as np 
from matplotlib import pyplot as plt 
import matplotlib
 
# fname 为 你下载的字体库路径，注意 SimHei.ttf 字体的路径
# zhfont1 = matplotlib.font_manager.FontProperties(fname="SimHei.ttf") 
 
x = np.arange(0,38)
# x=[k for k,v in match_score_pair_Huskies.items()]
# degree_centrality=[]
# clossness_centrality=[]
# betweenness_centrality=[]
# clustering=[]
# betweenness_centrality_2_best=[]
# plt.title("value_of_graph_per_game_2_players") 
 
# fontproperties 设置中文显示，fontsize 设置字体大小
plt.xlabel("match")
plt.ylabel("Score of Huskies")
# plt.plot(x,degree,label="degree") 
plt.plot(x,sorted(H_score_per_game),label="degree_centrality") 
plt.plot(x,[list_goal_num[i] for i in [k for k,v in match_score_pair_Huskies.items()]],label="degree_centrality") 

#plt.legend(["degree","degree_centrality","clossness_centrality",'betweenness_centrality',"clustering"])
#plt.legend(["Degree centrality","Clossness centrality",'Betweenness centrality*25',"Clustering/4"])
# plt.show()
plt.savefig("Score of Huskies.png")


# In[187]:


import numpy as np 
from matplotlib import pyplot as plt 
import matplotlib
 
# fname 为 你下载的字体库路径，注意 SimHei.ttf 字体的路径
# zhfont1 = matplotlib.font_manager.FontProperties(fname="SimHei.ttf") 
 
x = H_score_per_game
# x=[k for k,v in match_score_pair_Huskies.items()]
# degree_centrality=[]
# clossness_centrality=[]
# betweenness_centrality=[]
# clustering=[]
# betweenness_centrality_2_best=[]
# plt.title("value_of_graph_per_game_2_players") 
 
# fontproperties 设置中文显示，fontsize 设置字体大小
plt.xlabel("Score of Huskies")
plt.ylabel("Actual Goal")
# plt.plot(x,degree,label="degree") 
plt.scatter(x,[v for k,v in list_goal_diff.items()],label="Total Goal")
plt.scatter(x,[v for k,v in list_goal_num.items()],label="Total Goal")

#plt.legend(["degree","degree_centrality","clossness_centrality",'betweenness_centrality',"clustering"])
#plt.legend(["Degree centrality","Clossness centrality",'Betweenness centrality*25',"Clustering/4"])
# plt.show()
plt.savefig("Score of Huskies___.png")


# In[413]:


print(sort_dict(match_score_pair_Huskies,'key'))
a=[1,2,3]
list(reversed([v for k,v in sort_dict(match_score_pair_Huskies,'key').items()]))


# In[412]:


import numpy as np 
from matplotlib import pyplot as plt 
import matplotlib
 
# fname 为 你下载的字体库路径，注意 SimHei.ttf 字体的路径
# zhfont1 = matplotlib.font_manager.FontProperties(fname="SimHei.ttf") 
 
x = np.arange(0,38)

plt.xlabel("match")
plt.ylabel("Score")
# plt.plot(x,degree,label="degree") 
plt.plot(x,[v/10 for k,v in score_H_minus_O_sort_dict.items()],label="Huskies-Opponents Score/10") 
plt.plot(x,[list_goal_diff[i] for i in [k for k,v in match_score_pair_Huskies.items()]],label="Goal Difference") 
plt.plot(x,[list_goal_num[i] for i in [k for k,v in match_score_pair_Huskies.items()]],label="Goal Difference") 
plt.legend(["Huskies-Opponents Score/10","Goal Difference",'Total Goal'])
#plt.legend(["degree","degree_centrality","clossness_centrality",'betweenness_centrality',"clustering"])
#plt.legend(["Degree centrality","Clossness centrality",'Betweenness centrality*25',"Clustering/4"])
# plt.show()
plt.savefig("Relationship_between_scorediff&DG.png")


# In[314]:


[v for k,v in .items()]


# In[199]:


import numpy as np 
from matplotlib import pyplot as plt 
import matplotlib
 
# fname 为 你下载的字体库路径，注意 SimHei.ttf 字体的路径
# zhfont1 = matplotlib.font_manager.FontProperties(fname="SimHei.ttf") 
 
x = score_H_minus_O

plt.xlabel("score_H_minus_O")
plt.ylabel('DG/TG')
# plt.plot(x,degree,label="degree") 
# plt.plot(x,[v/10 for k,v in score_H_minus_O_sort_dict.items()],label="Huskies-Opponents Score/10")
# plt.legend()
plt.scatter(x,[v for k,v in list_goal_diff.items()],label="Goal difference") 
plt.scatter(x,[v for k,v in list_goal_num.items()],label="Total goal") 
plt.legend(["Goal difference",'Total Goal'])
#plt.legend(["degree","degree_centrality","clossness_centrality",'betweenness_centrality',"clustering"])
#plt.legend(["Degree centrality","Clossness centrality",'Betweenness centrality*25',"Clustering/4"])
# plt.show()
plt.savefig("Relationship_between_scorediff&DG_xy.png")


# In[423]:


#第一步：导入逻辑回归
from sklearn.linear_model import LogisticRegression
#第二步：创建模型：逻辑回归
model=LogisticRegression()
#第三步：训练模型
model.fit(np.array(score_H_minus_O).reshape(-1,1),[v for k,v in list_goal_num.items()])


# In[427]:


model.predict([[-1000],[2],[3]])


# In[ ]:


#截距
a=model.intercept_
#回归系数
b=model.coef_
#构建逻辑回归函数
def y_pred(x,y):
    #构建线性回归函数
    z=a+b[0,0]*x+b[0,1]*y
    #构建逻辑回归函数
    return 1/(1+np.exp(-z))


# In[429]:


model.intercept_


# In[ ]:


#截距
a=model.intercept_
#回归系数
b=model.coef_
#构建逻辑回归函数
def y_pred(x,y):
    #构建线性回归函数
    z=a+b[0,0]*x+b[0,1]*y
    #构建逻辑回归函数
    return 1/(1+np.exp(-z))


# In[223]:


from scipy.optimize import curve_fit

def sigmoid2(x, a,c, d):
    #y = a/ (1 + np.exp(-1*(x-d)))
    y=np.log(1+np.exp(-c*x+d))
    return y

xdata = score_H_minus_O
ydata = [v for k,v in list_goal_num.items()]

popt, pcov = curve_fit(sigmoid2, xdata, ydata)
print(popt)

x = np.arange(-38,38)#np.linspace(-50, 50, 50)
y = sigmoid2(x, *popt)

pylab.plot(xdata, ydata, 'o', label='data')
pylab.plot(x,y, label='fit')
#pylab.ylim(0, 1.05)
pylab.legend(loc='best')
pylab.show()


# In[226]:


import matplotlib.pyplot as plt
import numpy as np

# 生成数据
x = np.linspace(0, 20, 20)
y1 = np.random.randint(50, 100, 20)
y2 = np.random.randint(50, 100, 20)
y3 = np.random.randint(50, 100, 20)

# 堆积柱状图
plt.bar(x, y1, color='r', label='语文')
plt.bar(x, y2, bottom=y1, color='g', label='数学')
plt.bar(x, y3, bottom=y1+y2, color='c', label='英语')

# 显示范围
plt.xlim(-2, 22)
plt.ylim(0, 280)

# 添加图例
plt.legend(loc='upper right')
plt.grid(axis='y', color='gray', linestyle=':', linewidth=2)

plt.show()


# In[342]:


import numpy as np 
from matplotlib import pyplot as plt 
import matplotlib
 
 
x = list(df_3.index)
 
plt.xlabel("score_H_minus_O")
plt.ylabel('DG/TG')
# plt.plot(x,degree,label="degree") 
# plt.plot(x,[v/10 for k,v in score_H_minus_O_sort_dict.items()],label="Huskies-Opponents Score/10")
# plt.legend()
plt.scatter(x,np.array(df_3[1]),label="Goal difference") 
plt.scatter(x,np.array(df_3[2]),label="Total goal") 
plt.legend(["Goal difference",'Total Goal'])
#plt.legend(["degree","degree_centrality","clossness_centrality",'betweenness_centrality',"clustering"])
#plt.legend(["Degree centrality","Clossness centrality",'Betweenness centrality*25',"Clustering/4"])
# plt.show()
plt.savefig("Relationship_between_scorediff&DG_xy.png")


# In[323]:


df_3=pd.DataFrame([score_H_minus_O,[v for k,v in list_goal_diff.items()],[v for k,v in list_goal_num.items()]]).T


# In[324]:


df_3.head(5)


# In[336]:


df_3=df_3.groupby(0).mean()
df_3


# In[198]:


score_H_minus_O_sort_dict=sort_dict({i+1:score_H_minus_O[i] for i in range(0,38)})
score_H_minus_O_sort_dict



# In[540]:


win=list(H_score_per_game[[list(matches[matches['Outcome']=='win'].index)]])
tie=list(H_score_per_game[[list(matches[matches['Outcome']=='tie'].index)]])
loss=list(H_score_per_game[[list(matches[matches['Outcome']=='loss'].index)]])
avgwin=np.mean(win)
avgtie=np.mean(tie)
avgloss=np.mean(loss)
print(avgwin,avgtie,avgloss)
import numpy as np
import statsmodels.api as sm # recommended import according to the docs
import matplotlib.pyplot as plt
 
sample = win
ecdf1 = sm.distributions.ECDF(sample)
ecdf2 = sm.distributions.ECDF(tie)
ecdf3 = sm.distributions.ECDF(loss)
#等差数列，用于绘制X轴数据
x = np.linspace(35,63)
# x轴数据上值对应的累计密度概率
y1 = ecdf1(x)
y2 = ecdf2(x)
y3 = ecdf3(x)
#绘制阶梯图
plt.step(x, y1,label='win(EX=52.75)',linewidth=4)
plt.step(x, y2,label='tie(EX=50.29)',linewidth=4)
plt.step(x, y3,label='loss(EX=48,21)',linewidth=4)
plt.xlabel('Score')
plt.ylabel('ECDF of game')
plt.legend(['win(EX=52.75)','tie(EX=50.29)','loss(EX=48,21)'])
plt.show()
plt.savefig('ecdf.png')

import matplotlib.pyplot as plt
import numpy as np
T = x
#power1 = np.array([1.53E+03, 5.92E+02, 2.04E+02, 7.24E+01, 2.72E+01, 1.10E+01, 4.70E+00])

from scipy.interpolate import spline
xnew = np.linspace(T.min(),T.max(),300) #300 represents number of points to make between T.min and T.max
power_smooth = spline(T,y1,xnew)
power_smooth1 = spline(T,y2,xnew)
power_smooth2 = spline(T,y3,xnew)
plt.plot(xnew,power_smooth)
plt.plot(xnew,power_smooth1)
plt.plot(xnew,power_smooth2)
plt.plot(x,np.cumsum(y1)/sum(y1))
plt.show()


# In[487]:


win=list(np.array(score_H_minus_O)[[list(matches[matches['Outcome']=='win'].index)]])
tie=list(np.array(score_H_minus_O)[[list(matches[matches['Outcome']=='tie'].index)]])
loss=list(np.array(score_H_minus_O)[[list(matches[matches['Outcome']=='loss'].index)]])
avgwin=np.mean(win)
avgtie=np.mean(tie)
avgloss=np.mean(loss)
print(avgwin,avgtie,avgloss)
import numpy as np
import statsmodels.api as sm # recommended import according to the docs
import matplotlib.pyplot as plt
 
sample = win
ecdf1 = sm.distributions.ECDF(sample)
ecdf2 = sm.distributions.ECDF(tie)
ecdf3 = sm.distributions.ECDF(loss)
#等差数列，用于绘制X轴数据
x = np.linspace(-50,35)
# x轴数据上值对应的累计密度概率
y1 = ecdf1(x)
y2 = ecdf2(x)
y3 = ecdf3(x)
#绘制阶梯图
plt.step(x, y1,label='win(EX=11.389604832351818)',linewidth=4)
plt.step(x, y2,label='tie(EX=-4.02682154853765)',linewidth=4)
plt.step(x, y3,label='loss(EX=-15.053165490988189)',linewidth=4)
plt.xlabel('Score')
plt.ylabel('ECDF of game')
plt.legend(['win(EX=11.389604832351818)','tie(EX=-4.02682154853765)','loss(EX=-15.053165490988189)'])
plt.show()
plt.savefig('ecdf_2team_minus.png')


# In[541]:


from scipy.optimize import curve_fit

def sigmoid2(x, c, d):
    #y = a/ (1 + np.exp(-1*(x-d)))
    y=1/(1+np.exp(-c*(x-d)))
    return y

xdata = np.linspace(35,63)
ydata = ecdf1(np.linspace(35,63))
xdata2 = np.linspace(35,63)
ydata2 = ecdf2(np.linspace(35,63))
xdata3 = np.linspace(35,63)
ydata3 = ecdf3(np.linspace(35,63))
popt, pcov = curve_fit(sigmoid2, xdata, ydata,p0=(1,47))
print(popt)
popt2, pcov2 = curve_fit(sigmoid2, xdata2, ydata2,p0=(1,50))
print(popt2)
popt3, pcov3 = curve_fit(sigmoid2, xdata3, ydata3,p0=(1,53))
print(popt3)
x = np.arange(35,63)#np.linspace(-50, 50, 50)
y = sigmoid2(x, *popt)
y2 = sigmoid2(x, *popt2)
y3 = sigmoid2(x, *popt3)
pylab.step(xdata, ydata, label='Win(ECDF)')
pylab.plot(x,y, label='Win(Sigmoid)',linewidth=3)
pylab.step(xdata2, ydata2, label='Lie(ECDF)')
pylab.plot(x,y2, label='Tie(Sigmoid)',linewidth=3)
pylab.step(xdata3, ydata3, label='Loss(ECDF)')
pylab.plot(x,y3, label='Loss(Sigmoid)',linewidth=3)
#pylab.ylim(0, 1.05)
pylab.legend(loc='best')
plt.xlabel('Score')
plt.ylabel('ECDF of game')
pylab.show()


# In[539]:


from scipy.optimize import curve_fit

def sigmoid2(x, c, d):
    #y = a/ (1 + np.exp(-1*(x-d)))
    y=1/(1+np.exp(-c*(x-d)))
    return y

xdata = np.linspace(-50,35)
ydata = ecdf1(np.linspace(-50,35))
xdata2 = np.linspace(-50,35)
ydata2 = ecdf2(np.linspace(-50,35))
xdata3 = np.linspace(-50,35)
ydata3 = ecdf3(np.linspace(-50,35))
popt, pcov = curve_fit(sigmoid2, xdata, ydata,p0=(1,-15))
print(popt)
popt2, pcov2 = curve_fit(sigmoid2, xdata2, ydata2,p0=(1,-4))
print(popt2)
popt3, pcov3 = curve_fit(sigmoid2, xdata3, ydata3,p0=(1,11.3))
print(popt3)
x = np.arange(-50,35)#np.linspace(-50, 50, 50)
y = sigmoid2(x, *popt)
y2 = sigmoid2(x, *popt2)
y3 = sigmoid2(x, *popt3)
pylab.step(xdata, ydata, label='Win(ECDF)')
pylab.plot(x,y, label='Win(Sigmoid)',linewidth=3)
pylab.step(xdata2, ydata2, label='Lie(ECDF)')
pylab.plot(x,y2, label='Tie(Sigmoid)',linewidth=3)
pylab.step(xdata3, ydata3, label='Loss(ECDF)')
pylab.plot(x,y3, label='Loss(Sigmoid)',linewidth=3)
#pylab.ylim(0, 1.05)
pylab.legend(loc='best')
plt.xlabel('Score')
plt.ylabel('ECDF of game')
pylab.show()


# In[197]:


score_H_minus_O=[]
for id, row in matches.iterrows():
    score_H_minus_O.append(H_score_per_game[row['MatchID']-1]-team_score[row['OpponentID']])
score_H_minus_O


# In[399]:


team_score


# In[498]:


from SALib.sample import saltelli 
from SALib.analyze import sobol 
import matplotlib.pyplot as plt 

def ET(X): 
    # column 0 = x1, column 1 = x2, column 2 = x3 
    return(Q[0].real*X[:,0]+Q[1].real*X[:,1]+Q[2].real*X[:,2]+Q[3].real*X[:,3])

problem = {'num_vars': 3, 
      'names': ['x1', 'x2', 'x3'],
      'bounds': [[10, 100], 
        [3, 7], 
        [-10, 30]] 
      } 

# Generate samples 
param_values = saltelli.sample(problem, 1000, calc_second_order=False) 
print(param_values)
# Run model (example) 
Y = ET(param_values) 

# Perform analysis 
Si = sobol.analyze(problem, Y, print_to_console=True) 
# Print the first-order sensitivity indices
print (Si['S1'])

print (Si['ST'])
plt.subplots(figsize=(9, 9)) # 设置画面大小
plt.barh(range(len(Si['S1'])), Si['S1'])
plt.barh(range(len(Si['ST'])), Si['ST'])
plt.show()


# In[ ]:





# In[190]:


def sort_dict(dict_,type='value'):
    if type=='key':
        return dict(sorted(dict_.items(), key=lambda x: x[0], reverse=True))
    if type=='value':
        return dict(sorted(dict_.items(), key=lambda x: x[1], reverse=True))


# In[191]:


match_id=1
list_score={}
for i in H_score_per_game:
    list_score[match_id]=i
    match_id+=1
match_score_pair_Huskies=sort_dict(list_score)
match_score_pair_Huskies


# In[193]:


matches=pd.read_csv('matches.csv')
matches.head(5)


# In[194]:


list_goal_diff={}
for id,row in matches.iterrows():
    list_goal_diff[id+1]=row['OwnScore']-row['OpponentScore']
list_goal_diff


# In[195]:


list_goal_num={}
for id,row in matches.iterrows():
    list_goal_num[id+1]=row['OwnScore']
list_goal_num




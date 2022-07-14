#import and set data
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# DATA_PATH = "../../data/"
# data = pd.read_pickle(f"{DATA_PATH}df_filter_dummy_{TARGET_CITY}.pkl")

print("DATA LOADING........")
data_all = pd.read_pickle("../../data/pkl/merged_filtered_0630.pkl")
print("DONE")

TARGET_CITY = str(input("Which City You want to draw network?  ex)포항시   :  "))

#Get keyword set and preprocess
data = data_all[data_all['city'] == TARGET_CITY]
data.reset_index(inplace=True, drop = True)
print(f"{TARGET_CITY} START")
keyword_set = data['filter_keyword']

key_set = []
for item in keyword_set:
    item = str(item)
    item = item.replace("{", "")
    item = item.replace("}", "")
    item = item.replace("'", "")
    splited = item.split(",")
    temp = []
    for tok in splited:                         
        temp.append(tok.replace(" ", ""))
    key_set.append(temp)


#apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

te = TransactionEncoder()
te_result = te.fit(key_set).transform(key_set)
key_df = pd.DataFrame(te_result, columns=te.columns_)

itemset = apriori(key_df, min_support=0.001, use_colnames=True)

from mlxtend.frequent_patterns import association_rules
apriori_result = association_rules(itemset, metric="lift", min_threshold=1)

apriori_result.to_excel(f"./result/{TARGET_CITY}_apriori_result.xlsx")
print(f"apriori_result save as ./result/{TARGET_CITY}_apriori_result.xlsx")

from mlxtend.frequent_patterns import association_rules
apriori_result = association_rules(itemset, metric="lift", min_threshold=1)

#Network
network_data = apriori_result[['antecedents', 'consequents', 'lift']]

import matplotlib.pyplot as plt
import networkx as nx

node_lst = []
for node in network_data['antecedents']:
    node_lst.append(', '.join(list(node)))

print("If you don't have NanumGothic in your computer, It will rasie error here")
import matplotlib.font_manager as fm
import matplotlib

font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
fontprop = fm.FontProperties(fname=font_path, size=18)
font_name = fm.FontProperties(fname="/usr/share/fonts/truetype/nanum/NanumGothic.ttf").get_name()

matplotlib.rcParams['font.family'] ='NanumGothic'
matplotlib.rcParams['axes.unicode_minus'] =False

print(pd.DataFrame(network_data['lift']).quantile([0.5, 0.9, 0.99]))
thold = float(input("Set weight Threshold for graph : "))

keyword = 'keyword'
while True:
    keyword = str(input("Which Keyword you want to check? type 'q' to end program: "))
    if keyword == 'q':
        print("Close the Program")
        break

    graph = nx.Graph()
    child_node = []
    for i in range(len(network_data)):
        a, b, c = network_data.iloc[i, :]
        a = ', '.join(list(a))
        b = ', '.join(list(b))
        if a == keyword:
            if c > thold:
                graph.add_edge(a, b, weight = round(c,3))
                child_node.append(b)
        else:
            pass
    
    child_node2 = []
    for i in range(len(network_data)):
        a, b, c = network_data.iloc[i, :]
        a = ', '.join(list(a))
        b = ', '.join(list(b))
        if a in child_node:
            if c > thold:
                graph.add_edge(a, b, weight = round(c,3))
                child_node2.append(b)
        else:
            pass
        
    for i in range(len(network_data)):
        a, b, c = network_data.iloc[i, :]
        a = ', '.join(list(a))
        b = ', '.join(list(b))
        if a in child_node2:
            if c > thold:
                graph.add_edge(a, b, weight = round(c,3))
        else:
            pass
        

    try:
        edges,weights = zip(*nx.get_edge_attributes(graph,'weight').items())
    except:
        print(f"{keyword} is not in graph")
    else:
        n_size = dict(graph.degree)
        pos = nx.kamada_kawai_layout(graph)
        nx.draw(graph, nodelist=n_size.keys(), node_size=[v * 10 for v in n_size.values()], 
                with_labels = True, font_family = font_name, font_size = 2,
                alpha = 0.7,
                edgelist=edges, edge_color=weights, width=1, edge_cmap=plt.cm.binary)

        plt.title(f"{TARGET_CITY}의 {keyword} keyword network")
        #plt.show(block=False)
        plt.savefig(f"./program_graph/{TARGET_CITY}_{keyword}.png", format="PNG", dpi = 1000)
        print(f"Save {TARGET_CITY}_{keyword} well")
        plt.clf()

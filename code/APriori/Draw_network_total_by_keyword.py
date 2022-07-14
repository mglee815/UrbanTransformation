#import and set data
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# print("DATA LOADING........")
# keyword_set = pd.read_pickle("../../data/keyword_set.pkl")
# print("DONE")

# #apriori
# from mlxtend.preprocessing import TransactionEncoder
# from mlxtend.frequent_patterns import apriori

# te = TransactionEncoder()
# te_result = te.fit(keyword_set).transform(keyword_set)
# key_df = pd.DataFrame(te_result, columns=te.columns_)

# min_sup = 0.005
# print(f"apriori start with {min_sup} min support")
# itemset = apriori(key_df, min_support=min_sup, use_colnames=True)
# print("apriori done")

# def itemset_n(itemset, n):
#     for idx, item in enumerate(itemset.itemsets):
#         if len(item) == n:
#             idx
#             break 
#     itemset_n = itemset[:idx]
#     return itemset_n

# from mlxtend.frequent_patterns import association_rules
# apriori_result = association_rules(itemset_n(itemset, 5), metric="lift", min_threshold=1)

# apriori_result.to_excel(f"./result/total_apriori_result.xlsx")
# print("association_rule(lift) saved")

apriori_result = pd.read_excel("./result/total_apriori_result.xlsx")

def set_to_list(df_series):
    rtr_lst = []
    for item in df_series:
        item = str(item)
        item = item.replace("frozenset", "")
        item = item.replace("{", "")
        item = item.replace("}", "")
        item = item.replace("(", "")
        item = item.replace(")", "")
        item = item.replace("'", "")
        item = item.replace(" ", "")
        splited = item.split(",")
        temp = []
        for tok in splited:
            temp.append(tok.replace(" ", ""))
        rtr_lst.append(temp)

    return rtr_lst

apriori_result['antecedents'] = set_to_list(apriori_result['antecedents'])
apriori_result['consequents'] = set_to_list(apriori_result['consequents'])

#Network
network_data = apriori_result[['antecedents', 'consequents', 'lift']]

import matplotlib.pyplot as plt
import networkx as nx

node_lst = []
for node in network_data['antecedents']:
    node_lst.append(', '.join(list(node)))

#=========================================================================================================
print("If you don't have NanumGothic in your computer, It will rasie error here")
import matplotlib.font_manager as fm
import matplotlib

#Change font path fit to your enviroment
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
fontprop = fm.FontProperties(fname=font_path, size=18)
font_name = fm.FontProperties(fname="/usr/share/fonts/truetype/nanum/NanumGothic.ttf").get_name()

matplotlib.rcParams['font.family'] ='NanumGothic'
matplotlib.rcParams['axes.unicode_minus'] =False
#=========================================================================================================

print("distribution of lift")
print(pd.DataFrame(network_data['lift']).quantile([0.5, 0.9, 0.99]))
#thold = float(input("Set weight Threshold for graph : "))
thold = 1


keyword_list = pd.read_excel("../../data/Text_keyword_0620.xlsx")
keyword_list = keyword_list['Target 키워드']

for keyword in keyword_list:
        keyword = keyword.replace(" ", "")
    # while True:
    #     keyword = str(input("Which Keyword you want to check? type 'q' to end program: "))
    #     if keyword == 'q':
    #         print("Close the Program")
    #         break

        graph = nx.DiGraph()
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
            for key in pos.keys():
                pos[key] *= 100
            nx.draw(graph, nodelist=n_size.keys(), pos = pos,
                    with_labels = True, font_family = font_name, font_size = 2,
                    alpha = 0.7,
                    edgelist=edges, edge_color=weights, width=1, edge_cmap=plt.cm.binary)

            plt.title(f"전체 도시의 '{keyword}' keyword network")
            plt.savefig(f"./program_graph/direct/Total_{keyword}_Direct_lift{thold}.png", format="PNG", dpi = 1000)
            print(f"Save 'Total_{keyword}_Direct_lift{thold}.png' well")
            plt.clf()

#import and set data
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# DATA_PATH = "../../data/"
# TARGET_CITY = str(input("Which City You want to draw network?  "))
# data = pd.read_pickle(f"{DATA_PATH}df_filter_dummy_{TARGET_CITY}.pkl")


data_all = pd.read_pickle("../../data/pkl/merged_filtered_0630.pkl")

print("If you don't have NanumGothic in your computer, It will rasie error here")
import matplotlib.font_manager as fm

font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
fontprop = fm.FontProperties(fname=font_path, size=18)

import matplotlib.font_manager as fm 
from matplotlib import rc
font_name = fm.FontProperties(fname="/usr/share/fonts/truetype/nanum/NanumGothic.ttf").get_name()

import matplotlib

matplotlib.rcParams['font.family'] ='NanumGothic'
matplotlib.rcParams['axes.unicode_minus'] =False

for TARGET_CITY in data_all['city'].unique():
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

    for idx, item in enumerate(itemset.itemsets):
        if len(item) == 3:
            idx
            break
    itemset_2gram = itemset[:idx]


    from mlxtend.frequent_patterns import association_rules
    apriori_result = association_rules(itemset_2gram, metric="lift", min_threshold=1)

    apriori_result.to_excel(f"./result/{TARGET_CITY}_apriori_result.xlsx")
    print(f"apriori_result save as ./result/{TARGET_CITY}_apriori_result.xlsx")

    from mlxtend.frequent_patterns import association_rules
    apriori_result = association_rules(itemset_2gram, metric="lift", min_threshold=1)

    #Network
    network_data = apriori_result[['antecedents', 'consequents', 'lift']]

    import matplotlib.pyplot as plt
    import networkx as nx

    node_lst = []
    for node in network_data['antecedents']:
        node_lst.append(', '.join(list(node)))

    graph = nx.Graph()
    #graph.add_nodes_from(node_lst)
    for i in range(len(network_data)):
        a, b, c = network_data.iloc[i, :]
        a = ', '.join(list(a))
        b = ', '.join(list(b))
        if c > 1:
            graph.add_edge(a, b, weight = round(c,3))
        else:
            pass

    labels = nx.get_edge_attributes(graph,'weight')

    edges,weights = zip(*nx.get_edge_attributes(graph,'weight').items())



    weights = list(weights)
    weights[np.argmax(weights)] = 9

    def weights_multiple(weights, n):
        lst = []
        for w in weights:
            lst.append(w * n)
        return lst

    weights_m = weights_multiple(weights, 1)


    n_size = dict(graph.degree)
    pos = nx.shell_layout(graph)

    nx.draw(graph, nodelist=n_size.keys(), node_size=[v * 10 for v in n_size.values()], 
            with_labels = True, font_family = font_name, font_size = 5,
        alpha = 0.7,
        edgelist=edges, edge_color=weights_m, width=1, edge_cmap=plt.cm.binary)

    plt.title("태백시의 keyword network")
    #plt.show(block=False)
    plt.savefig(f"./result/{TARGET_CITY}.png", format="PNG", dpi = 1000)
    print(f"Save {TARGET_CITY} well")
    plt.clf()

print("DONE")

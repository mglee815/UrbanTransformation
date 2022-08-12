#import and set data
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# DATA_PATH = "../../data/"
# data = pd.read_pickle(f"{DATA_PATH}df_filter_dummy_{TARGET_CITY}.pkl")

#==========get data=========================================
print("DATA LOADING........")
cat_arti_mat = np.loadtxt("../../result/topic_idx.csv", delimiter=",", dtype = np.int8)
data_all = pd.read_csv("../../data/topic_data.csv")
print("DONE")

#TARGET_CITY = str(input("Which City You want to draw network?  ex)포항시   :  "))
TARGET_CITY = '포항시'



print("If you don't have NanumGothic in your computer, It will rasie error here")
import matplotlib.font_manager as fm
import matplotlib

font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
fontprop = fm.FontProperties(fname=font_path, size=18)
font_name = fm.FontProperties(fname="/usr/share/fonts/truetype/nanum/NanumGothic.ttf").get_name()

matplotlib.rcParams['font.family'] ='NanumGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

for topic in range(8):
    print(f"Topic {topic +1} START")
    data_topic = data_all.iloc[cat_arti_mat[topic] == 1, :]

#========filter by city and separate into three data set======================
    data = data_topic[data_topic['city'] == TARGET_CITY]
    data.reset_index(inplace=True, drop = True)
    print(f"{TARGET_CITY} START")

    data1 = data[data['일자'] < 20000000]
    data1.reset_index(inplace=True, drop = True)
    data2 = data[data['일자'] < 20100000]
    data2 = data2[data2['일자'] > 20000000]
    data2.reset_index(inplace=True, drop = True)
    data3 = data[data['일자'] > 20100000]
    data3.reset_index(inplace=True, drop = True)

    print(f"{topic+1} / total : {data.shape}, ")
    print(f"{topic+1} / period1 : {data1.shape}, ")
    print(f"{topic+1} / period2 : {data2.shape}, ")
    print(f"{topic+1} / period3 : {data3.shape}, ")

    data_lst = [data, data1, data2, data3]


    #========for loop for three period===============
    for period, dataset in enumerate(data_lst):
        print(f"==========C{TARGET_CITY}_T{topic + 1}_P{period} START============")
        keyword_set = dataset['filter_keyword']
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

        #==================apriori=====================================
        #apriori
        from mlxtend.preprocessing import TransactionEncoder
        from mlxtend.frequent_patterns import apriori

        te = TransactionEncoder()
        te_result = te.fit(key_set).transform(key_set)
        key_df = pd.DataFrame(te_result, columns=te.columns_)

        itemset = apriori(key_df, min_support=0.01, use_colnames=True)

        from mlxtend.frequent_patterns import association_rules
        apriori_result = association_rules(itemset, metric="confidence", min_threshold=0.5)

        apriori_result.to_excel(f"../result/C{TARGET_CITY}_T{topic+1}_P{period}_apriori_result.xlsx")
        print(f"apriori_result save as ../result/C{TARGET_CITY}_T{topic+1}_P{period}_apriori_result.xlsx")


        #===============Draw Network=======================================
        network_data = apriori_result[['antecedents', 'consequents', 'lift']]
        print(f"netword_data : {network_data.shape}")

        import matplotlib.pyplot as plt
        import networkx as nx

        node_lst = []
        for node in network_data['antecedents']:
            node_lst.append(', '.join(list(node)))

        qtl = pd.DataFrame(network_data['lift']).quantile([0.5, 0.7, 0.9, 0.99])
        print(qtl)
        #thold = qtl.iloc[1,0]
        thold = 1.5

        #===============Build graph===================================
        graph = nx.DiGraph()
        child_node = []
        for i in range(len(network_data)):
            a, b, c = network_data.iloc[i, :]
            a = ', '.join(list(a))
            b = ', '.join(list(b))
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
            
        
        print("Draw Net")
        #===========Check edges and weights=====================
        try:
            edges,weights = zip(*nx.get_edge_attributes(graph,'weight').items())
        except:
            print(f"error")
        else:
            n_size = dict(graph.degree)
            pos = nx.kamada_kawai_layout(graph)
            nx.draw(graph, nodelist=n_size.keys(), 
                    #node_size=[ (v*5) + 50 for v in n_size.values()],
                    with_labels = True, font_family = font_name, font_size = 2,
                    alpha = 0.5,
                    edgelist=edges, edge_color=weights, width=0.5, edge_cmap=plt.cm.binary)

            plt.title(f"{TARGET_CITY}시 {topic+1}토픽의 {period}시기 keyword network")
            #plt.show(block=False)
            plt.savefig(f"../program_graph/city_topic_period/C{TARGET_CITY}_T{topic + 1}_P{period}.png", format="PNG", dpi = 1000)
            print(f"Save ../program_graph/city_topic_period/C{TARGET_CITY}_T{topic + 1}_P{period} well")
            plt.clf()

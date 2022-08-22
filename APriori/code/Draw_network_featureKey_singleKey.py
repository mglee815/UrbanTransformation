#import and set data
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# DATA_PATH = "../../data/"
# data = pd.read_pickle(f"{DATA_PATH}df_filter_dummy_{TARGET_CITY}.pkl")

#==========get data=========================================
print("DATA LOADING........")
cat_arti_mat = np.loadtxt("../../result/topic_idx.csv", delimiter=",", dtype = np.int8)
data_all = pd.read_csv("../../data/topic_data.csv")
keyword_list = pd.read_excel("../../data/Text_keyword_0620.xlsx")
keyword_list = list(keyword_list['Target 키워드'])

cat_lst = [
    ['발생', '피해'],
    ['건설', '유치', '인프라'],
    ['개최', '관광', '체험', '국제'],
    ['활성화', '지역경제'],
    ['인구', '감소'],
    ['도심'],
    ['산업'],
    ['주민']
]

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

        print(f"====================C{TARGET_CITY}_T{topic + 1}_P{period} START======================")
        keyword_set = dataset['특성추출(가중치순 상위 50개)']
        key_set = []
        for item in keyword_set:
            item = str(item)
            splited = item.split(",")
            temp = []
            for tok in splited:                         
                temp.append(tok.replace(" ", ""))
            key_set.append(temp)
        
        if len(key_set) == 10:
            break

        print("Keyword_set_prepared")


        #==================apriori=====================================
        #apriori
        from mlxtend.preprocessing import TransactionEncoder
        from mlxtend.frequent_patterns import apriori

        te = TransactionEncoder()
        te_result = te.fit(key_set).transform(key_set)
        key_df = pd.DataFrame(te_result, columns=te.columns_)

        itemset = apriori(key_df, min_support=0.1, use_colnames=True)
        print('itemset_built')
        from mlxtend.frequent_patterns import association_rules
        apriori_result = association_rules(itemset, metric="confidence", min_threshold=0.5)
        apriori_result.to_csv(f"../program_graph/city_topic_period/CTP_network_0822_featureKey/apriori_raw/C{TARGET_CITY}_T{topic+1}_P{period}_apriori.csv")
        print(f"apriori_result save as ../program_graph/city_topic_period/CTP_network_0822_featureKey/apriori_raw/C{TARGET_CITY}_T{topic+1}_P{period}_apriori.csv")

        idx_lst = []
        i = 0
        for a, b in zip(apriori_result['antecedents'], apriori_result['consequents']):
            if len(a) == 1:
                if len(b) == 1:
                    idx_lst.append(i)
            i += 1

        print(f"# of single keyword set : {len(idx_lst)}")
        apriori_single = apriori_result.iloc[idx_lst,:]
        
        #===============Draw Network=======================================
        network_data = apriori_single[['antecedents', 'consequents', 'lift']]
        print(f"netword_data : {network_data.shape}")

        import matplotlib.pyplot as plt
        import networkx as nx

        thold = 1
        qtl = pd.DataFrame(network_data['lift']).quantile([0.5, 0.7, 0.9, 0.99])
        #print(qtl)

        #===============Build graph===================================
        network_data = network_data[network_data['lift'] > thold]
        graph = nx.DiGraph()
        for i in tqdm(range(len(network_data))):
            a, b, c = network_data.iloc[i, :]
            a = list(a)
            b = list(b)
            a = a[0].replace(" ", "")
            b = b[0].replace(" ", "")
            graph.add_edge(a, b, weight = round(c,3))

        print(f"Number of node : {graph.number_of_nodes()}")

        key_match_lst = []
        i = 0
        for node in graph.nodes:
            if node in keyword_list:
                key_match_lst.append(node)
        print("Draw Net")
        #===========Check edges and weights=====================
        try:
            edges,weights = zip(*nx.get_edge_attributes(graph,'weight').items())
        except:
            print(f"It is empty graph")
        else:
            n_size = dict(graph.degree)
            pos = nx.kamada_kawai_layout(graph)
            nx.draw(graph, nodelist=n_size.keys(), 
                    #node_size=[ (v*5) + 50 for v in n_size.values()],
                    with_labels = True, font_family = font_name, font_size = 5,
                    alpha = 0.7,
                    pos = pos,
                    node_color = 'grey',
                    edgelist=edges, edge_color=weights, width=1, edge_cmap=plt.cm.binary)

            nx.draw_networkx_nodes(
                graph.subgraph(key_match_lst),
                pos = pos,
                node_color = 'blue'
            )
            nx.draw_networkx_nodes(
                graph.subgraph(cat_lst[topic]),
                pos = pos,
                node_color = 'red'
            )

            plt.title(f"{TARGET_CITY}시 {topic+1}토픽의 {period}시기 keyword network")
            #plt.show(block=False)
            plt.savefig(f"../program_graph/city_topic_period/CTP_network_0822_featureKey/C{TARGET_CITY}_T{topic + 1}_P{period}.png", format="PNG", dpi = 1000)
            print(f"Save ../program_graph/city_topic_period/CTP_network_0822_featureKey/C{TARGET_CITY}_T{topic + 1}_P{period} well")
            plt.clf()

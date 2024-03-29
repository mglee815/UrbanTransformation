#import and set data
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
import stopword_pre
warnings.filterwarnings('ignore')

#==========get data=========================================

#########################수정 필요#######################
#####################################데이터가 저장된 경로#################
print("DATA LOADING........")
cat_arti_mat = np.loadtxt("../data/topic_idx.csv", delimiter=",", dtype = np.int8)
data_all = pd.read_csv("../data/topic_data.csv")
keyword_list = pd.read_excel("../data/Text_keyword_0620.xlsx")
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

print("....Data loading finish")

TARGET_CITY_LIST = ['포항시', '상주시', '태백시', '광명시', '목포시', '통영시']
TARGET_CITY_LIST = ['포항시']


######################수정 필요#############
########################################

# 한국어로 그래프를 표시하기 위하여 NanumGothic을 컴퓨터에 설치하는 작업이 필요합니다.
# 만약 이전에 한국어로 pyplot을 사용한 경험이 없다면 주석처리 해도 되는 작업입니다.
print("If you don't have NanumGothic in your computer, It will rasie error here")
import matplotlib.font_manager as fm
import matplotlib

#다운로드한 나눔고딕을 설치한 이후에 그 경로를 명시해주면 됩니다.
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
fontprop = fm.FontProperties(fname=font_path, size=18)

# 경로 명시
font_name = fm.FontProperties(fname="/usr/share/fonts/truetype/nanum/NanumGothic.ttf").get_name()

matplotlib.rcParams['font.family'] ='NanumGothic'
matplotlib.rcParams['axes.unicode_minus'] = False


#첫번째로 Target city에 대한 반복문
for TARGET_CITY in TARGET_CITY_LIST:
    data_len_lst = []
    #두번째로 topic에 대한 반복문
    for topic in range(8):
        print(f"Topic {topic +1} START")
        data_topic = data_all.iloc[cat_arti_mat[topic] == 1, :]

    #========filter by city and separate into three data set======================
        data = data_topic[data_topic['city'] == TARGET_CITY]
        data.reset_index(inplace=True, drop = True)
        print(f"{TARGET_CITY} START")

        #세번째 기준인 날짜별로 데이터를 분활
        data1 = data[data['일자'] < 20000000]
        data1.reset_index(inplace=True, drop = True)
        data2 = data[data['일자'] < 20100000]
        data2 = data2[data2['일자'] > 20000000]
        data2.reset_index(inplace=True, drop = True)
        data3 = data[data['일자'] > 20100000]
        data3.reset_index(inplace=True, drop = True)

        data_lst = [data, data1, data2, data3]
        data_len = [len(data), len(data1), len(data2), len(data3)]
        data_len_lst.append(data_len)
        print(f"data_len : {data_len_lst}")


        #========for loop for three period===============
        #마지막으로 기간별로 반복문
        for period, dataset in enumerate(data_lst):

            print(f"====================C{TARGET_CITY}_T{topic + 1}_P{period} START======================")
            keyword_set = dataset['특성추출(가중치순 상위 50개)']
            key_set = []
            for item in keyword_set:
                item = str(item)
                item = stopword_pre.stopword(item)
                splited = item.split(",")
                temp = []
                for tok in splited:                         
                    temp.append(tok.replace(" ", ""))
                key_set.append(temp)
            print(len(key_set))
            if len(key_set) <= 30:
                print("too small keyset")
                print(len(key_set))
                continue

            print("Keyword_set_prepared")


            #==================apriori=====================================
            #apriori
            from mlxtend.preprocessing import TransactionEncoder
            from mlxtend.frequent_patterns import apriori

            te = TransactionEncoder()
            te_result = te.fit(key_set).transform(key_set)
            key_df = pd.DataFrame(te_result, columns=te.columns_)

            ####################기준치 설정###################################
            ##############################################################
            itemset = apriori(key_df, min_support=0.1, use_colnames=True)
            print('itemset_built')
            from mlxtend.frequent_patterns import association_rules

            ########################################기준치 설정#################################
            ##############################################################
            apriori_result = association_rules(itemset, metric="confidence", min_threshold=0.3)

            ###########수정필요#############
            ##################################
            apriori_result.to_csv(f"{apriorio 결과를 저장하고 싶은 경로}/C{TARGET_CITY}_T{topic+1}_P{period}_apriori.csv")
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
            #print(f"netword_data : {network_data.shape}")

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

                ###########################수정필요##############################
                ##############################################################
                plt.savefig(f"{그래프를 저장하고 싶은 경로}/C{TARGET_CITY}_T{topic + 1}_P{period}.png", format="PNG", dpi = 1000)
                print(f"Save {그래프를 저장하고 싶은 경로}/C{TARGET_CITY}_T{topic + 1}_P{period} well")
                plt.clf()
        
    pd.DataFrame(data_len_lst).to_csv(f"{기사 수를 저장하고 싶은 경로}/{TARGET_CITY}_article_num.csv")
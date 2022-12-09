import pandas as pd
import datetime

w = int(input("window size: "))
m = int(input("min size: "))

start_time = datetime.datetime.now()
print("data load start")
DATA_PATH = "../data/pkl/"
data = pd.read_pickle(f"{DATA_PATH}merged_filtered_0620.pkl") #currunt
#data = pd.read_pickle(f"{DATA_PATH}merged_filtered.pkl") #previos
print("loaded")
print(datetime.datetime.now() - start_time)


#kileed 됨
#key_lst = data['키워드'].apply(lambda x : x.split(','))

start_time = datetime.datetime.now()
print('split start')
key_lst = []
for i, atcl in enumerate(data['특성추출(가중치순 상위 50개)']):
    key_lst.append(str(atcl).split(','))
print("splited")
print(datetime.datetime.now() - start_time)


print('model build')
from gensim.models import Word2Vec
model = Word2Vec(sentences=key_lst, window=w, min_count=m, workers=8, sg=0)

from gensim.models import KeyedVectors
model.save(f'../result/w2v_model/{datetime.datetime.now().strftime("%m_%d_%H")}_w2v_w{w}m{m}') # 모델 저장
print('model saved')
print(datetime.datetime.now() - start_time)

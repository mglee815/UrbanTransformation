import pandas as pd
import datetime

w = int(input("window size: "))
m = int(input("min size: "))

start = int(input("from : "))
end = int(input("to : "))

start_time = datetime.datetime.now()
print("data load start")
DATA_PATH = "../data/pkl/"
data = pd.read_pickle(f"{DATA_PATH}merged_filtered_0620.pkl") #currunt
#data = pd.read_pickle(f"{DATA_PATH}merged_filtered.pkl") #previos
print("loaded")
print(datetime.datetime.now() - start_time)


y = []
def str_to_dt(x):
    temp = datetime.datetime.strptime(str(x), '%Y%m%d')
    y.append(temp.year)

data['일자'].apply(lambda x : str_to_dt(x))
data['year'] = y



temp = data[data['year'] >= start]
temp.reset_index(inplace=True, drop = True)
data = temp[temp['year'] <= end]
print(data.info())

start_time = datetime.datetime.now()
print('split start')
key_lst = []
for i, atcl in enumerate(data['특성추출(가중치순 상위 50개)']):
    key_lst.append(str(atcl).split(','))
print("splited")
print(datetime.datetime.now() - start_time)



#for idx, txt in enumerate(list(data['특성추출(가중치순 상위 50개)'])):
#     if type(txt) == float:
#         data.iloc[idx, 15] = "NONE"


print('model build')
from gensim.models import Word2Vec
model = Word2Vec(sentences=key_lst, window=w, min_count=m, workers=8, sg=0)

from gensim.models import KeyedVectors
MODEL_NAME = f'{datetime.datetime.now().strftime("%m_%d_%H")}_w2v_{start}_{end}_w{w}m{m}'
model.save(f'../result/w2v_model/{MODEL_NAME}') # 모델 저장
print(f'model saved  as {MODEL_NAME}')
print(datetime.datetime.now() - start_time)

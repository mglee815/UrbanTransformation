import pandas as pd

print("start")
DATA_PATH = "../data/pkl/"
data = pd.read_pickle(f"{DATA_PATH}merged_filtered.pkl")
print("loaded")

#kileed 됨
#key_lst = data['키워드'].apply(lambda x : x.split(','))

key_lst = []
for i, atcl in enumerate(data['특성추출(가중치순 상위 50개)']):
    key_lst.append(atcl.split(','))

from gensim.models import Word2Vec
model = Word2Vec(sentences=key_lst, window=30, min_count=10, workers=8, sg=0)

from gensim.models import KeyedVectors
model.wv.save_word2vec_format('../result/w2v_model/total_w2v_w30m10') # 모델 저장
#loaded_model = KeyedVectors.load_word2vec_format('../result/w2v_model/sample_w2v')
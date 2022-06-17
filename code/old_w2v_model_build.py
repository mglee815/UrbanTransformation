import pandas as pd

print("start")
DATA_PATH = "../data/pkl/"
data1 = pd.read_pickle(f"{DATA_PATH}splited1.pkl")
data2 = pd.read_pickle(f"{DATA_PATH}splited2.pkl")
print("loaded")

data1
#===========================================================#
##############키워드 수가 너무 많아서 특성 키워드로만 해보기############

# from gensim.models import Word2Vec
# model = Word2Vec(sentences=key_lst, window=5, min_count=50, workers=8, sg=0)

# from gensim.models import KeyedVectors
# model.wv.save_word2vec_format('../result/w2v_model/total_w2v_w30m10') # 모델 저장
# #loaded_model = KeyedVectors.load_word2vec_format('../result/w2v_model/sample_w2v')
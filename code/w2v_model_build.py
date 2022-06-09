import pandas as pd

data = pd.read_pickle("../data/merged_data/total/merged_filtered.pkl")
key_lst = data['키워드'].apply(lambda x : x.split(','))

from gensim.models import Word2Vec
model = Word2Vec(sentences=key_lst, window=15, min_count=5, workers=8, sg=0)

from gensim.models import KeyedVectors
model.wv.save_word2vec_format('../result/w2v_model/sample_w2v') # 모델 저장
#loaded_model = KeyedVectors.load_word2vec_format('../result/w2v_model/sample_w2v')
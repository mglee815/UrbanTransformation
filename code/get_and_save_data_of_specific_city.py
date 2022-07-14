import pandas as pd

df = pd.read_pickle("../data/df_filter_dummy.pkl")
print("Data loaded")

TARGET_CITY = str(input("Type city you want to extract"))

temp = df[df['city'] == TARGET_CITY]
temp.to_pickle(f"../data/df_filter_dummy_{TARGET_CITY}.pkl")
print("Save Well")
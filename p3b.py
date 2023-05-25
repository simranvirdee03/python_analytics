import numpy as np
import pandas as pd
from sklearn import linear_model
from word2number import w2n
import math
import joblib
df=pd.read_csv("C:/Users/Harpreet Singh/Desktop/ml csv files/hiring.csv")

df.experience=df.experience.fillna("zero")
df.experience=df.experience.apply(w2n.word_to_num)

med=math.floor(df['test_score(out of 10)'].mean())
print(med)
df['test_score(out of 10)']=df['test_score(out of 10)'].fillna(med)
print(df)
reg=linear_model.LinearRegression()
reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])
print("First Question",reg.predict([[2,9,6]]))
print("Second Question",reg.predict([[12,10,10]]))
joblib.dump(reg,'P3b')
joblib.load('P3b')
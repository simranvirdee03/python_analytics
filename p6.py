import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
df = pd.read_csv("C:/Users/Harpreet Singh/Desktop/ml csv files/homeprice2.csv")

dummies = pd.get_dummies(df.town)
merged = pd.concat([df, dummies], axis='columns')
final = merged.drop(['town', 'west windsor'], axis='columns')
print(final)

reg = linear_model.LinearRegression()
X = final.drop(['price'], axis='columns')
Y = final['price']
reg.fit(X, Y)
print(reg.predict([[2800,0,1]]))
print(reg.score(X,Y))

le=LabelEncoder()
dfle=df
dfle.town=le.fit_transform(dfle.town)

X=dfle[['town','area']].values
Y=dfle['price'].values

ct=ColumnTransformer([('town',OneHotEncoder(),[0])],remainder='passthrough')
X=ct.fit_transform(X)

X=X[:,1:]
print(X)
reg.fit(X,Y)
print(reg.predict([[1,0,2800]]))
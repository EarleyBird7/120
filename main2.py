import pandas as pd
from sklearn import datasets
wine = deatasets.load_wine()
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X= wine.data
y=wine.target


x_train1,x_test1,y_train1,y_test1= train_test_split(X,y,test_size=0.25,random_state=42)

sc= StandardScaler()

x_train1= sc.fit_transform(x_train1)
x_test1= sc.fit_transform(x_test1)

model1= GaussianNB()
model1.fit(x_train1,y_train1)

y_pred1= model1.predict(x_test1)

accuracy= accuracy_score(y_test1,y_pred1)

print(accuracy)

x_train2,x_test2,y_train2,y_test2= train_test_split(X,y,test_size=0.25,random_state=42)

sc= StandardScaler()

x_train2= sc.fit_transform(x_train2)
x_test2= sc.fit_transform(x_test2)

model2=LogisticRegression(random_state=0)
model2.fit(x_train2,y_train2)

y_pred2= model2.predict(x_test2)

accuracy2= accuracy_score(y_test2,y_pred2)

print(accuracy2)
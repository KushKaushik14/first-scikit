import numpy as np 
import pandas as pd 
heart_disease = pd.read_csv("heart-disease.csv")
X = heart_disease.drop("target",axis=1) #features 
Y = heart_disease["target"] #labels 
from sklearn.ensemble import RandomForestClassifier # type: ignore
clf = RandomForestClassifier()
params = clf.get_params()
from sklearn.model_selection import train_test_split #type: ignore
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
clf.fit(X_train,Y_train)
tscore = clf.score(X_train, Y_train)
print(tscore)
score = clf.score(X_test,Y_test)
print(score)





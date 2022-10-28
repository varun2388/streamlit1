import pandas as pd
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load

claimants = pd.read_csv("claimants.csv")
claimants.drop(["CASENUM"],inplace=True,axis = 1)
claimants = claimants.dropna()

X = claimants.iloc[:,[1,2,3,4,5]]
Y = claimants.iloc[:,0]
model = LogisticRegression()
model.fit(X,Y)

dump(model,open('Logistic_Model.sav', 'wb'))

loaded_model=load(open('Logistic_Model.sav' ,'rb'))
result = loaded_model.score(X,Y)
print(result)
 
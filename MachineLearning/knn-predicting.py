from sklearn import preprocessing
import pandas as pd
import pickle

# Dataset to predict
player = {'minutes': [1818,1819,1320], 'goals': [9,5,9], 'assists': [1,4,1],'cleansheets':[2,9,2],'yc':[7,4,2]}
df = pd.DataFrame(data=player)
print(df.head())

#
MinMaxScaler = preprocessing.MinMaxScaler()
X_data_minmax = MinMaxScaler.fit_transform(df)
data = pd.DataFrame(X_data_minmax,columns=['minutes', 'goals', 'assists', 'cleansheets','yc'])
print(data)
with open('model_pickle','rb') as f:
    mp=pickle.load(f)
prediction = mp.predict(data)
for i in prediction:
    print('prediction ',i)

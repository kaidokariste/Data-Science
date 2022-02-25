from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

#Import data
df = pd.read_csv('ml-premierl.csv',sep=';')
print(df.head())

# Look over unique classes
print(df['class'].unique())

#Look over null values
print(df.isnull().values.any())

#Map football positions as numbers where DEF = 1, MID = 2 and FWD = 3
df['class'] = df['class'].map({'DEF' :1, 'MID' :2, 'FWD' :3}).astype(int)
print(df.head())

# ## lets do EDA (Exploratory Data Analysis)
# plt.close()
# sns.set_style('whitegrid')
# sns.pairplot(df, hue='class', height=3)
# plt.show()
#
# sns.set_style('whitegrid')
# sns.FacetGrid(df, hue='class', height=5).map(plt.scatter, 'goals', 'cleansheets').add_legend()
# plt.show()
# #Minutes and goals are best way to determine player type

#Normalization
x_data = df.drop(['class'], axis=1)
y_data = df['class']
MinMaxScaler = preprocessing.MinMaxScaler()
X_data_minmax = MinMaxScaler.fit_transform(x_data)
data = pd.DataFrame(X_data_minmax,columns=['minutes', 'goals', 'assists', 'cleansheets','yc'])
print(data.head())

#Define model
knn_clf=KNeighborsClassifier(n_neighbors = 3)

## Training the model
# Divide the data into into training and and test datasets
X_train, X_test, y_train, y_test = train_test_split(data, y_data,test_size=0.2, random_state = 1)
knn_clf.fit(X_train,y_train)

#These are the predicted output values
predictions=knn_clf.predict(X_test)
comparison = pd.DataFrame({'Real':y_test, 'Predictions':predictions})
print(comparison)

#Probability or accuracy how the test data was predicted
print(knn_clf.score(X_test,y_test))



# saving model
# https://www.youtube.com/watch?v=KfnhNlD8WZI&ab_channel=codebasics

#with open('model_pickle','wb') as f:
#    pickle.dump(knn_clf, f)
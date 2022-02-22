## Supervised Learning with scikit-learn
### k-nearest neighbours
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(iris['data'], iris['target'])
iris['data'].shape # Result(150.4) , what is 150 rows and 4 columns
iris['target'].shape # (150,), has one column with 150 rows
```
Predicting

```python
X_new = np.array([[5.6, 2.8, 3.9, 1.1],[5.7, 2.6, 3.8, 1.3],[4.7, 3.2, 1.3, 0.2]])
prediction = knn.predict(X_new)
X_new.shape
print('Prediction: {}'.format(prediction)) # Prediction: [1 1 0]

```

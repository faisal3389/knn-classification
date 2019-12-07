import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from time import time


dataframe = pd.read_csv('./sample_data/patient_demographics.csv')
print(dataframe)

X_old =  dataframe.drop('TARGET_CLASS' , axis=1)

# X = scale(X_old)
X = X_old

y = dataframe['TARGET_CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


error_rate = []

for i in range(1,15):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


plt.figure(figsize=(10,4))
plt.plot(range(1,15), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K-Values')
plt.xlabel('K-Values')
plt.ylabel('Error Rate')
plt.show()

optimal_k_index = error_rate.index(min(error_rate))

optimal_k_value = optimal_k_index + 1

knn = KNeighborsClassifier(n_neighbors=optimal_k_value)
knn.fit(X_train, y_train)
start_time = time()
pred = knn.predict(X_test)
end_time = time()
seconds_elapsed = end_time - start_time
print(seconds_elapsed)
prob = knn.predict_proba(X_test)

print(y_test)
print(pred)
print(prob)
print('k-NN accuracy for test set: %f' % knn.score(X_test, y_test))

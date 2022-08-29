import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler

train_label = []
train_samples = []

for i in range(1000):
    rand_young = randint(13, 64)
    train_samples.append(rand_young)
    train_label.append(0)

    rand_old = randint(65, 100)
    train_samples.append(rand_old)
    train_label.append(1)

for i in range(50):
    rand_young = randint(13, 64)
    train_samples.append(rand_young)
    train_label.append(1)

    rand_old = randint(65, 100)
    train_samples.append(rand_old)
    train_label.append(0)


train_samples = np.array(train_samples)
train_label = np.array(train_label)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))
scaled_test_samples = scaler.fit_transform(np.array([14, 15, 16, 89, 78]).reshape(-1,1))
# print(scaled_train_samples)


#lets classify
from sklearn import svm
clf_svm = svm.SVC()
clf_svm.fit(scaled_train_samples, train_label)

print(clf_svm.predict(scaled_test_samples))
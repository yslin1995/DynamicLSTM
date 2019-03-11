import numpy as np
import cPickle as pickle


all_train = [] # the first 10 numbers are inputs, the last one is label in each row
all_test = []
for _ in range(100000):
    a = np.random.choice(10, 100)
    b = np.random.choice(10, 1)
    c = a[b+90]
    train = np.append(np.append(a,b),c)
    all_train.append(train)


for _ in range(100000):
    a = np.random.choice(10, 100)
    b = np.random.choice(10, 1)
    c = a[b+90]
    test = np.append(np.append(a, b),c)
    all_test.append(test)

# print(all_test)
# print(all_train)
pickle.dump(all_train,open('num_train10_1.pkl','wb'))
pickle.dump(all_test,open('num_test10_1.pkl','wb'))


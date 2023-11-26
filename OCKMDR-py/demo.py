from OCKMDR import *
 
import numpy as np
import time 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn import svm
import scipy.io as sio

import warnings
warnings.filterwarnings('ignore')


# Parameters
gamma = 0.1



#### Loading from a file *.csv
#x_train_path = r'..\test-data\x-train.csv'
#y_train_path = r'..\test-data\y-train.csv'
#x_test_path  = r'..\test-data\x-test.csv'
#y_test_path  = r'..\test-data\y-test.csv'

### Training data
#y_train, x_train = read_data_csv(x_train_path, y_train_path)
## Ratio of the number of objects of the class 
#nu = round(np.sum(y_train == -1) / len(y_train), 4)

### Testing data
#y_test, x_test = read_data_csv(x_test_path, y_test_path)



### Loading from a file *.mat
train_path = r"..\test-data\OCmodel003_data.mat"
test_path = r"..\test-data\OCmodel004_data.mat"

## Training data
y_train, x_train = read_data_mat(train_path, "X", "Y")
# Ratio of the number of objects of the class 
nu = round(np.sum(y_train == -1) / len(y_train), 4)

## Testing data
y_test, x_test = read_data_mat(test_path, "X", "Y")



### SVM solution
print(f"SVM params: -g {gamma} -n {nu}")

start_time = time.time()
algoSVM = svm.OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
modelSVM = algoSVM.fit(x_train)
end_time = time.time()

print(f"Elapsed time: {end_time - start_time:.4f} sec")

p_label = modelSVM.predict(x_test)
p_scores = modelSVM.score_samples(x_test)
print(quality(y_test, np.array(p_label)))
print("AUC (labels): ", round(roc_auc_score(y_test, p_label), 4))
print("AUC (scores): ", round(roc_auc_score(y_test, p_scores), 4))


### KMDR solution
nrs = 10; srs = 100
print(f"\nKMRD params: -nrs {nrs} -srs {srs}")

start_time = time.time()
algoKMDR = OneClassKMDR(nu=nu, gamma=gamma, nrs=nrs, srs=srs)
modelKMDR = algoKMDR.fit(x_train)
end_time = time.time()

print(f"Elapsed time: {end_time - start_time:.4f} sec")

p_label, p_scores = modelKMDR.predict(x_test)
print(quality(y_test, np.array(p_label)))
print("AUC (labels): ", round(roc_auc_score(y_test, p_label), 4))
print("AUC (scores): ", round(roc_auc_score(y_test, p_scores), 4))
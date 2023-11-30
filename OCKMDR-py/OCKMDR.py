import os, csv
import numpy as np
from sklearn import svm
import scipy.io as sio


# assert(isinteger(expertY) && isinteger(classifierY))
def quality(expertY, classifierY):
    TP = np.sum((expertY ==  1) ==  classifierY)
    TN = np.sum((expertY == -1) == -classifierY)
    FN = np.sum((expertY ==  1) == -classifierY)
    FP = np.sum((expertY == -1) ==  classifierY)

    if (TP + FN != 0):
        Recall = TP / (TP + FN)
    else:
        Recall = 0

    if (TP + FP != 0):
        Precision = TP / (TP + FP)
    else:
        Precision = 0
    
    if (Precision + Recall != 0):
        Fmeasure = round(2 * Precision * Recall / (Precision + Recall), 5)
    else:
        Fmeasure = 0

    return {"TP":TP, "TN":TN, "FP":FP, "FN":FN, "Fmeasure":Fmeasure}


# 
def read_data_csv(x_path, y_path):
    with open(x_path, 'r') as __file:
        reader = csv.reader(__file, delimiter = ',')
        x_train = np.array(list(reader), dtype = float)
      
    with open(y_path, 'r') as __file:
        reader = csv.reader(__file, delimiter = ',')
        y_train = np.array(list(reader), dtype = float)
    
    return y_train.ravel(), x_train


#
def read_data_mat(path, name_X, name_Y):
    mat_contents = sio.loadmat(path)
    return mat_contents[name_Y].ravel(), mat_contents[name_X]


#
def rbf_kernal(_obj1, _obj2, _gamma):
    return np.exp(-1 * _gamma * np.linalg.norm(_obj2 - _obj1) ** 2)


class OneClassKMDR:
    def __init__(self, nrs, srs, nu, gamma):
        self._nrs = nrs
        self._srs = srs
        self._nu = nu
        self._gamma = gamma


    def fit(self, _data):
        n_objects = _data.shape[0]
    
        self.rho = 0
        self.sv_coef = np.zeros(n_objects)
        self.submodels = []

        # training, random subsamples
        for i in range(0, self._nrs):
            tempIndex = np.random.choice(n_objects, self._srs, replace = False)            
            tempData = _data[tempIndex, :]
        
            tempModel = svm \
                            .OneClassSVM(nu = self._nu, \
                                         gamma = self._gamma, \
                                         kernel = "rbf") \
                            .fit(tempData)

            offsetIndex = tempIndex[tempModel.support_]
            self.sv_coef[offsetIndex] = self.sv_coef[offsetIndex] + tempModel.dual_coef_
            self.rho = self.rho + tempModel.offset_

            self.submodels.append(tempModel)

        # averaging
        self.sv_coef = self.sv_coef / self._nrs
        self.rho = self.rho / self._nrs
    
        nonzeroIndex = self.sv_coef.astype(bool)
        self.sv_coef = self.sv_coef[nonzeroIndex]
        self.sv = _data[nonzeroIndex]

        return self


    def predict(self, _data):
        n_objects = _data.shape[0]
        labels = np.zeros(n_objects, dtype=int)
        scores = np.zeros(n_objects, dtype=float)

        for i, x in enumerate(_data):
            for sv_coef, sv in zip(self.sv_coef, self.sv):
                scores[i] = scores[i] + sv_coef * rbf_kernal(sv, x, self._gamma)
            scores[i] = scores[i] - self.rho
            labels[i] = +1 if scores[i] > 0.0 else -1

        return labels, scores

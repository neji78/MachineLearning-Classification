import numpy as np
def sum(arr):
    r = 0
    for i in arr:
        r += i
    return r
def mean(data):
    x = data[0]
    y = data[1]
    r = data[2]
    n = len(r)
    k = sum(r)
    p1 = 0
    p2 = 0
    for i in range(n):
        p1 += x[i]*r[i]
        p2 += y[i]*r[i]
    return [[p1/k],[p2/k]]
def cov(data):
    x = data[0]
    y = data[1]
    r = data[2]
    n = len(r)
    k = sum(r)
    p1 = 0
    p2 = 0
    p3 = 0
    p4 = 0
    for i in range(n):
        p1 += (x[i]*x[i])*r[i]
        p2 += (x[i]*y[i])*r[i]
        p3 += (y[i]*x[i])*r[i]
        p4 += (y[i]*y[i])*r[i]
    return [[p1/k,p2/k],[p3/k,p4/k]]

def Pbar(data):
    r = data[2]
    k = sum(r)
    return k/len(r)
def calculate_bigWi(invcov):
    return -1/2*invcov
def calculate_wi(invcov,mean):
    return invcov*mean
def calculate_wi0(invc,mean,cova,pbar):
    mt = np.transpose(np.array(mean))
    m = np.array(mean)
    detCov = np.linalg.det(np.matrix(cova))
    f1 = -1/2*mt*invc*m
    f2 = -1/2*np.log(detCov)
    f3 = np.log(pbar)
    return f1 + f2 + f3
def error(oR,eR):
    return sum((np.array(oR) - np.array(eR))**2)/len(oR)
def findClass(arr):
    r = list()
    for i in range(len(arr)):
        elem = arr[i]
        if elem > -2.0:
            r.append(1)
        else:
            r.append(0)
    return np.array(r)
class Classifier:
    def __init__(self,lData):
        self.ldata = lData
        _mean = mean(self.ldata)
        _cov = cov(self.ldata)
        _Pbar = Pbar(self.ldata)
        invcov = np.linalg.inv(np.matrix(_cov))
        self.bigWi = calculate_bigWi(invcov)
        self.wi = calculate_wi(invcov,_mean)
        self.wi0 = calculate_wi0(invcov,_mean,_cov,_Pbar)
    def g(self,data):
        X = data[0]
        Y = data[1]
        R = data[2]
        gArr = list()
        for i in range(len(X)):
            x = np.array([[X[i]],[Y[i]]])
            xt = np.matrix.transpose(x)
            wit = np.matrix.transpose(self.wi)
            gi = xt*self.bigWi*x+wit*x+self.wi0
            glist = gi.tolist()
            gArr.append(glist[0][0])
        ri = findClass(gArr)
        return [gArr,ri],error(np.array(R),ri)
        
    
        
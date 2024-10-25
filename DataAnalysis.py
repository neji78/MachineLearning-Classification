import numpy as np
import JsonMaker as jm
import classificator as cl
import os
def mean(data):
    sum = 0
    m = list()
    for arr in data:
        m.append(np.mean(arr))
    return m

def var(data):
    v = list()
    for arr in data:
        v.append(np.var(arr))
    return v
def cov(data):
    X = data[0]
    Y = data[1]
    M = mean(data)
    mx = M[0]
    my = M[1]
    sum = 0
    N = len(X)
    for i in range(N):
        xi = X[i]
        yi = Y[i]
        sum += (xi - mx)*(yi - my)
    return sum/N

def rho(varMatrix):
    row1 = varMatrix[0]
    row2 = varMatrix[1]
    return row1[1]/(row1[0]**0.5*row2[1]**0.5)

def calculateErrors(eParam,oParam):
    mError = ((np.array(oParam["mean"]) - np.array(eParam["mean"]))**2)/2
    covError = ((np.matrix(oParam["covariance matrix"]) - np.matrix(eParam["covariance matrix"]))**2)/2
    corError = ((oParam["corrolation"] - eParam["corrolation"])**2)/2
    errors = dict()
    errors["mean error"] = mError.tolist()
    errors["covariance error"] = covError.tolist()
    errors["corrolation error"] = corError
    return errors
    
def extractParameters(data):
    parameters = dict()
    parameters["mean"] = mean(data)
    variance = var(data)
    covariance = cov(data)
    covMatrix = [[variance[0],covariance],[covariance,variance[1]]]
    corrolation = rho(covMatrix)
    parameters["covariance matrix"] = covMatrix
    parameters["corrolation"] = corrolation
    return parameters

def analysisData(dir,op):
    xdata = np.loadtxt(dir+"/data.txt")
    data = [xdata[0],xdata[1]]
    pr = jm.Json(dir+"/info")
    ep = extractParameters(data) #estimated parmeters
    pr.add("Estimated Parameters",ep)
    err = calculateErrors(ep,op)
    pr.add("Errros",err)
    pr.save()

def analysis(root,oParam):
    learnDir = root + "/learn"
    testDir = root + "/test"
    oParam["corrolation"] = rho(oParam["covariance matrix"])
    analysisData(learnDir,oParam)
    analysisData(testDir,oParam)
    
    learnData = np.loadtxt(learnDir+"/data.txt")
    testData = np.loadtxt(testDir+"/data.txt")
    classifier = cl.Classifier(learnData)
    result,err = classifier.g(testData)
    if not os.path.isdir(root+"/classification/"):
        os.mkdir(root+"/classification/")
    np.savetxt(root+"/classification/data.txt",result)
    pr1 = jm.Json(root+"/classification/error")
    pr1.add("err",err)
    pr1.save()



import numpy as np
import math
import os
def supervised_class(x,y):
    r = list()
    n = len(x)
    for i in range(n):
        xi = x[i]
        yi = y[i]
        if xi*yi > 0.5:
            r.append(1)
        else:
            r.append(0)
    return r
class DataGenerator:
    def __init__(self,mean,cov,dir):
        self.mean = mean
        self.cov = cov
        self.dir = dir
    def generate(self,n):
       if os.path.isdir(self.dir):
           return
       dirs = self.dir.split("/")
       s = ""
       for i in dirs:
           s += i
           if not os.path.isdir(s):
            os.mkdir(s)
           s += "/"
       os.mkdir(s+"learn")
       os.mkdir(s+"test")
       x,y = np.random.multivariate_normal(self.mean, self.cov, n).T 
       ldcursor = int(math.floor(n*0.75))
       r = np.array(supervised_class(x,y))
       ld = [x[:ldcursor],y[:ldcursor],r[:ldcursor]]
       td = [x[ldcursor:],y[ldcursor:],r[ldcursor:]]
       np.savetxt(self.dir+"/learn/data.txt",ld)
       np.savetxt(self.dir+"/test/data.txt",td)

# print(X[0])
# plt.plot(X[0], X[1], 'x')

# plt.axis('equal')
# 
# plt.show()
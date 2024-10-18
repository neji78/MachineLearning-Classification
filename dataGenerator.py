import numpy as np
import matplotlib.pyplot as plt
class DataGenerator:
    def __init__(self,mean,cov):
        self.mean = mean
        self.cov = cov
    def generate(self):
       x,y = np.random.multivariate_normal(self.mean, self.cov, 5000).T 
       ld = [x[:4000],y[:4000]]
       td = [x[4000:],y[4000:]]
       np.savetxt("learningData.txt",ld)
       np.savetxt("testData.txt",td)

# print(X[0])
# plt.plot(X[0], X[1], 'x')

# plt.axis('equal')
# 
# plt.show()
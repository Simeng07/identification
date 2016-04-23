import numpy

p1 = numpy.load("predy1.npy")
p2 = numpy.load("predy2.npy")
p3 = numpy.load("predy3.npy")
p0 = p1 + p2 + p3
real = numpy.load("predyy.npy")

error1 = (p1.argmax(axis=1) == real).sum()
error2 = (p2.argmax(axis=1) == real).sum()
error3 = (p3.argmax(axis=1) == real).sum()
error0 = (p0.argmax(axis=1) == real).sum()

print float(error1)/real.shape[0]
print float(error2)/real.shape[0]
print float(error3)/real.shape[0]
print float(error0)/real.shape[0]

from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X_tsne = TSNE().fit_transform(p1)
plt.figure()
s = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=real, s=numpy.pi*10)
cbar = plt.colorbar(s, ticks=[0,1,2,3,4,5,6,7,8])
cbar.ax.set_yticklabels(['aigo','Allbar','HYUNDAI','JWD','LG','OLYMPUS','PHILIPS','Shinco','SONY'])
plt.show()

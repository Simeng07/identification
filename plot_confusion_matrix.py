"""
================
Confusion matrix
================

Example of confusion matrix usage to evaluate the quality
of the output of a classifier on the iris data set. The
diagonal elements represent the number of points for which
the predicted label is equal to the true label, while
off-diagonal elements are those that are mislabeled by the
classifier. The higher the diagonal values of the confusion
matrix the better, indicating many correct predictions.

The figures show the confusion matrix with and without
normalization by class support size (number of elements
in each class). This kind of normalization can be
interesting in case of class imbalance to have a more
visual interpretation of which class is being misclassified.

Here the results are not as good as they could be as our
choice for the regularization parameter C was not the best.
In real life applications this parameter is usually chosen
using :ref:`grid_search`.

"""

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

# import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=0.01)
y_pred = classifier.fit(X_train, y_train).predict(X_test)


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title, fontsize=20)

    cb = plt.colorbar()
    for t in cb.ax.get_yticklabels(): 
        t.set_fontsize(15) 

    tick_marks = np.arange(len(iris.target_names))
    plt.xticks(np.arange(9), ('aigo','Allbar','HYUNDAI','JWD','LG','OLYMPUS','PHILIPS','Shinco','SONY'), rotation=45, fontsize=15)
    plt.yticks(np.arange(9), ('aigo','Allbar','HYUNDAI','JWD','LG','OLYMPUS','PHILIPS','Shinco','SONY'), fontsize=15)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)


# Compute confusion matrix
#cm = confusion_matrix(y_test, y_pred)
cm = np.asarray([[ 99,0,0, 0 ,  0  , 0  , 0  , 1,  0],[  0 , 95 ,  0  , 0  , 1  , 0 ,  0  , 4  , 0], [  2  , 0 , 92  , 0  , 0  , 0  , 0  , 6  , 0], [  0 ,  0 ,  1 , 95 ,  0,   2 ,  2  , 0 ,  0], [  1 ,  0 ,  1 ,  0 , 89 ,  0 ,  0  , 1 ,  8], [  1 ,  0 ,  0 , 17 ,  0 , 82 ,  0 ,  0 ,  0], [  0  , 0 ,  0 ,  0 ,  0 ,  0 ,100 ,  0 ,  0], [  0 , 19 ,  0 ,  0 ,  0 ,  0 ,  0 , 81 ,  0], [  0 ,  0  , 0,   0 ,  0 ,  0 ,  0 ,  0 ,100]])
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
#plt.figure()
#plot_confusion_matrix(cm)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
print(1-cm_normalized)
plt.figure()
#plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
plot_confusion_matrix(1-cm_normalized, title='Normalized confusion matrix', cmap=plt.cm.hot)

plt.show()

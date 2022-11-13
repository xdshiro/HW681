import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

xNeg = multivariate_normal.rvs(size=50, mean=[-5, -5], cov=[[5, 0], [0, 5]], random_state=4)
yNeg = np.ones(50) * -1
xPos = multivariate_normal.rvs(size=50, mean=[5, 5], cov=[[5, 0], [0, 5]], random_state=5)
yPos = np.ones(50)
X = np.concatenate((xNeg, xPos), axis=0)
y = np.concatenate((yNeg, yPos), axis=0)

from sklearn.svm import SVC



# print(clf.coef_)
# print(clf.intercept_)
# print(clf.support_vectors_)
def z(x, y, coeff, inter):
    return x * coeff[0][0] + y * coeff[0][1] + inter

def plotting_boundary(clf, title):
    fig = plt.figure()
    fig.set_size_inches(10, 5)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)
    zArray = np.ones(len(clf.support_vectors_))
    zArray[0:len(zArray) // 2] -= 2
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], zArray, marker='o', c='r', s=100)
    ax.scatter(X[:, 0], X[:, 1], y, marker='o')
    xArray = np.linspace(-10, 10, 100)
    yArray = np.linspace(-10, 10, 100)
    xx, yy = np.meshgrid(xArray, yArray)
    ax.plot_surface(xx, yy, -z(xx, yy, clf.coef_, clf.intercept_), alpha=0.2)
    ax2.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], marker='o', c='r', s=130)
    ax2.scatter(xNeg[:, 0], xNeg[:, 1])
    ax2.scatter(xPos[:, 0], xPos[:, 1], color='g')
    coef = clf.coef_[0]
    y_array = -coef[0] / coef[1] * xArray - (clf.intercept_[0]) / coef[1]
    ax2.plot(xArray, y_array)
    print(f'Number of support vectors: {len(clf.support_vectors_[:, 0])} for C={C}')
    fig.suptitle(title, fontsize=16)
    plt.show()

C = 1
clf = SVC(C=0.001, kernel='linear', random_state=0)
clf.fit(X, y)
print('Accuracy: ', clf.score(X, y))
plotting_boundary(clf, f'C={C}')
exit()


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from chefboost import Chefboost as chef
iris = datasets.load_iris()
# print(iris.feature_names, iris.target_names)
# exit()
# print(iris)
# exit()
X = iris.data
y = iris.target
print(X)
print(y)
exit()
# print(X)
# print(y)
dTrain = pd.read_csv('carseats_train.csv', names=None)
dTest = pd.read_csv('carseats_test.csv')
clf = DecisionTreeClassifier(random_state=1234, criterion='entropy', splitter='best')
X = dTrain.iloc[:, 1:3]
y = dTrain['Sales']
model = clf.fit(X, y)
print(model)
text_representation = tree.export_text(clf)
print(text_representation)
print(dTrain)
# with open("decistion_tree.log", "w") as fout:
#     fout.write(text_representation)
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf,
                   feature_names=iris.feature_names,
                   class_names=iris.target_names,
                   filled=True)
# dataValues[1:10]
fig.show()
# fig.savefig("decistion_tree.png")
def H(p):
    return -p*np.log(p) - (1-p)* np.log(1-p)

def error(p):
    return p

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    exit()
    ticksFontSize = 20
    xyLabelFontSize = 20
    fig, ax = plt.subplots(figsize=(8, 6))
    xArray = np.linspace(0.0001, 0.5, 200)

    plt.plot(xArray, H(xArray), lw=5, color='b')
    plt.plot(xArray, error(xArray), lw=5, color='r')
    plt.xticks(fontsize=ticksFontSize)
    plt.yticks(fontsize=ticksFontSize)
    ax.set_xlabel('p', fontsize=xyLabelFontSize)
    # ax.set_ylabel(yname, fontsize=xyLabelFontSize)
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

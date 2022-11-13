import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
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

clf = DecisionTreeClassifier(random_state=1234)
# X = dTrain.iloc[:, 1:3]
# y = dTrain['Sales']
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
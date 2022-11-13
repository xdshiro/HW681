import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

dTrain = pd.read_csv('carseats_train.csv', names=None)

clf = DecisionTreeClassifier(random_state=2131, criterion='entropy', splitter='best')

A=[1, 1, 1, 1, 0, 0, 0, 0]
B=[1, 1, 0, 0, 1, 1, 0, 0]
C=[1, 0, 1, 0, 1, 0, 1, 0]
y=[1, 1, 1, 1, 1, 0, 0, 0]
X = np.array([A,B,C]).T
print(X)
model = clf.fit(X, y)
print(model)
text_representation = tree.export_text(clf)
print(text_representation)
# print(dTrain)
# with open("decistion_tree.log", "w") as fout:
#     fout.write(text_representation)
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf,
                   feature_names=['A', 'B', 'C'],
                   class_names=['0', '1'],
                   filled=True)
# dataValues[1:10]
fig.show()
# fig.savefig("decistion_tree.png")
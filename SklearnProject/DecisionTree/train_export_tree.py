from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
iris = load_iris()
petal_l_w = iris.data[:,2:]
labels = iris.target
tree_clf = DecisionTreeClassifier(max_depth=2)
clf = tree_clf.fit(petal_l_w, labels)
if not os.path.exists("./images"):
    os.makedirs("./images")
'''Export train parameters.'''
export_graphviz(
    tree_clf,
    out_file="./images/iris_tree.dot",
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)
'''Use command in terminal: dot -Tpng ./images/iris_tree.dot -o ./images/iris_tree.png'''
'''show images.'''
image_raw = mpimg.imread("./images/iris_tree.png")
plt.figure(figsize=(8, 8))
plt.imshow(image_raw)
plt.show()

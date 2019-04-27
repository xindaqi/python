from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.font_manager import FontProperties
font = FontProperties(fname="/usr/share/fonts/truetype/arphic/ukai.ttc")
iris = load_iris()
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02
petal_l_w = iris.data[:,2:]
labels = iris.target
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(petal_l_w, labels)
# print("petal length and width: {}".format(petal_l_w))
x_min, x_max = petal_l_w[:,0].min() - 1, petal_l_w[:, 0].max() + 1
y_min, y_max = petal_l_w[:, 1].min() - 1, petal_l_w[:, 1].max() + 1
print("x min: {},x max: {}".format(x_min, x_max))
print("y min: {},y max: {}".format(y_min, y_max))
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                    np.arange(y_min, y_max, plot_step))
print("xx: {} \n yy: {}".format(xx, yy))
print("shape of xx: {} \n shape of yy: {}".format(xx.shape, yy.shape))
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
z = tree_clf.predict(np.c_[xx.ravel(), yy.ravel()])
print("z: {}".format(z))
print("shape of z: {}".format(z.shape))
z = z.reshape(xx.shape)
print("reshape z: {}".format(z))
cs = plt.contourf(xx, yy, z, cmap=plt.cm.RdYlBu)
# plt.figure(figsize=(6, 6))
makers = ["^", "s", "p"]
flower_name = ["山鸢尾", "杂色鸢尾", "维吉尼亚鸢尾"]
for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(labels == i)
    plt.scatter(petal_l_w[idx, 0], petal_l_w[idx, 1], c=color, 
                label=iris.target_names[i]+flower_name[i], cmap=plt.cm.RdYlBu, 
                edgecolor='black', s=60, marker=makers[i])
plt.xlabel("花瓣长度/cm",fontproperties=font)
plt.ylabel("花瓣宽度/cm",fontproperties=font)
plt.title("决策分区", fontproperties=font)
plt.legend(prop=font,loc="lower right")
plt.savefig("./images/partition.png", format="png")
plt.show()

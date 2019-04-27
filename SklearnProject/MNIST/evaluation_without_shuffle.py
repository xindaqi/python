from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname="/usr/share/fonts/truetype/arphic/ukai.ttc")

import numpy as np
import os
TRAIN_STEPS = 10
mnist = fetch_mldata("MNIST original", data_home="./datasets")
images, labels = mnist["data"], mnist["target"]
train_images, train_labels, test_images, test_labels = images[:60000], labels[:60000], images[60000:], labels[60000:]
# train_images, train_labels, test_images, test_labels = images[:6], labels[:6], images[60000:], labels[60000:]
# shuffle_index = np.random.permutation(6)
# shuffle index: [5 3 1 2 0 4]
# print("shuffle index: {}".format(shuffle_index))
'''Create shuffle index for mix the train data'''
# shuffle_index = np.random.permutation(60000)
# train_images, train_labels = train_images[shuffle_index], train_labels[shuffle_index]
'''Get True or false whether label equal 5 or not.'''
train_labels_5 = (train_labels == 5)
# [False False False False False False]
# print("train labels 5: {}".format(train_label_5))
# print("Type of labels: {}".format(type(train_label_5)))
test_labels_5 = (test_labels == 5)
'''Create classifier'''
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(train_images, train_labels_5)
test_image = images[36000]
pre_res = sgd_clf.predict([test_image])
print("Predicted result: {}".format(pre_res))
cross_value = cross_val_score(sgd_clf, train_images, train_labels_5, cv=3, scoring="accuracy")
print("Modle prediction accuracy: {}".format(cross_value))
predict_labels = cross_val_predict(sgd_clf, train_images, train_labels_5, cv=3)
print("predict labels: {}".format(predict_labels))
print("shape of predict labels: {}".format(predict_labels.shape))
cfm = confusion_matrix(train_labels_5, predict_labels)
print("confusion matrix: {}".format(cfm))
precision_value = precision_score(train_labels_5, predict_labels)
print("Precision of model predition: {}".format(precision_value))
recall_value = recall_score(train_labels_5, predict_labels)
print("Recall of model prediction: {}".format(recall_value))
f1_value = f1_score(train_labels_5, predict_labels)
print("F1 score: {}".format(f1_value))
predict_values = cross_val_predict(sgd_clf, train_images, train_labels_5, cv=3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(train_labels_5, predict_values)
if not os.path.exists("./images"):
    os.makedirs("./images")
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.figure(figsize=(10, 4))
    plt.plot(thresholds, precisions[:-1], "b--", label="精度")
    plt.plot(thresholds, recalls[:-1], "g-", label="召回率")
    plt.xlabel("阈值", fontproperties=font)
    plt.legend(loc="upper left", prop=font)
    plt.ylim([0, 1])
    plt.grid("on")
    plt.xlim([-1500000, 600000])
    plt.savefig("./images/pre_recall_threshold.png", format="png")
    plt.show()
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
def plot_precision_recall(precisions, recalls):
    plt.figure(figsize=(6, 6))
    plt.plot(recalls[:-1], precisions[:-1], "g-")
    plt.xlabel("召回率", fontproperties=font)
    plt.ylabel("精度", fontproperties=font)
    plt.ylim([0, 1])
    plt.grid("on")
    plt.xlim([0, 1])
    plt.savefig("./images/precision_recall.png", format="png")
    plt.show()
plot_precision_recall(precisions, recalls)
'''ROC'''
fpr, tpr, thresholds = roc_curve(train_labels_5, predict_values)
def plot_roc_curve(fpr, tpr, label=None):
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel("假正类率", fontproperties=font)
    plt.ylabel("真正类率", fontproperties=font)
    plt.grid("on")
    plt.savefig("./images/fpr_tpr.png", format="png")
    plt.show()
plot_roc_curve(fpr, tpr)
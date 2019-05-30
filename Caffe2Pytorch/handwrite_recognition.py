#coding:utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import operator
import caffe2.python.predictor.predictor_exporter as pe
# from caffe2.python import core
from caffe2.python import brew, core, model_helper, net_drawer, optimizer, visualize, workspace
from IPython import display
# import requests, StringIO, zipfile
# import urllib2, urllib

# caffe2初始化细节:caffe2_log_level=1
core.GlobalInit(["caffe2", "--caffe2_log_level=1"])
USE_LENET_MODEL = True
# 当前文件夹路径
current_folder = os.path.abspath(os.path.dirname(__name__))
# 图像数据集路径
data_folder = os.path.join(current_folder, "turorial_data", "mnist")
# 根路径,存储模型日志文件,及workspace分配的空间
root_folder = os.path.join(current_folder, "turorial_files", "tutorial_mnist")
db_missing = False
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
if os.path.exists(os.path.join(data_folder, "mnist-train-nchw-lmdb")):
    print("lmdb train db found!")
else:
    print("Please download datasets manually!")

workspace.ResetWorkspace(root_folder)
print("training data folder:" + data_folder)
print("workspace root folder:" + root_folder)

def AddInput(model, batch_size, db, db_type):
    '''数据读取
	参数:
		model: 模型结构
		batch_size: 数据组尺寸
		db: lmdb数据
		db_type: 数据格式
	返回:
		data: 图像数据
		label: 图像标签
	'''
    data_uint8, label = model.TensorProtosDBInput([], ["data_uint8", "label"], batch_size=batch_size, db=db, db_type=db_type)
    data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    data = model.Scale(data, data, scale=float(1./256))
    data = model.StopGradient(data, data)
    return data, label

def AddMLPModel(model, data):
    '''普通神经网络模型.
	参数:
		model: 模型结构
		data: 图像数据
	返回:
		softmax: 类别数据
	'''
    size = 28*28*1
    sizes = [size, size*2, size*2, 10]
    layer = data
    for i in range(len(sizes)-1):
        layer = brew.fc(model, layer, 'dense_{}'.format(i), dim_in=sizes[i], dim_out=sizes[i+1])
        layer = brew.relu(model, layer, 'relu_{}'.format(i))
    softmax = brew.softmax(model, layer, 'softmax')
    return softmax

def AddLeNetModel(model, data):
    '''卷积神经网络.
	参数:
		model: 模型结构
		data: 图像数据
	返回:
		softmax: 类别数据
	'''
    conv1 = brew.conv(model, data, 'conv1', dim_in=1, dim_out=20, kernel=5)
    pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
    conv2 = brew.conv(model, pool1, 'conv2', dim_in=20, dim_out=50, kernel=5)
    pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
    fc3 = brew.fc(model, pool2, 'fc3', dim_in=50*4*4, dim_out=500)
    relu3 = brew.relu(model, fc3, 'relu3')
    pred = brew.fc(model, relu3, 'pred', dim_in=500, dim_out=10)
    softmax = brew.softmax(model, pred, 'softmax')
    return softmax

def AddModel(model, data):
    '''选择模型结构.
	参数:
		model: 模型结构
		data: 图像数据
	返回:
		softmax: 对应模型的分类数据
	'''
    if USE_LENET_MODEL:
        return AddLeNetModel(model, data)
    else:
        return AddMLPModel(model, data)

def AddAccuracy(model, softmax, label):
    '''模型准确率.
	参数:
		model: 模型结构
		softmax: 分类数据
		lable:图像标签
	返回:
		accuracy: 识别准确率
	'''
    accuracy = brew.accuracy(model, [softmax, label], 'accuracy')
    return accuracy

def AddTrainingOperators(model, softmax, label):
    '''优化参数,训练模型.
	参数:
		model: 模型结构
		softmax: 分类数据
		label: 图像标签
	返回:
		None
	'''
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    loss = model.AveragedLoss(xent, 'loss')
    AddAccuracy(model, softmax, label)
    model.AddGradientOperators([loss])
    optimizer.build_sgd(model, base_learning_rate=0.1, policy="step", stepsize=1, gamma=0.999)

def AddBookkeepingOperator(model):
    '''日志记录,保存到文件或日志,本程序使用root_folder作为workspace,
	可在root_folder找到.
	参数:
		model: 模型结构
	返回:
		None
	'''
    model.Print("accuracy", [], to_file=1)
    model.Print("loss", [], to_file=1)
    for param in model.params:
        model.Summarize(param, [], to_file=1)
        model.Summarize(model.param_to_grad[param], [], to_file=1)
# 图像数据格式
arg_scope = {"order":"NCHW"}
# 模型结构
train_model = model_helper.ModelHelper(name="mnist_train", arg_scope=arg_scope)
data, label = AddInput(train_model, batch_size=64, 
                      db=os.path.join(data_folder, "mnist-train-nchw-lmdb"),
                      db_type='lmdb')
softmax = AddModel(train_model, data)
AddTrainingOperators(train_model, softmax, label)
AddBookkeepingOperator(train_model)
test_model = model_helper.ModelHelper(name="mnist_test", arg_scope=arg_scope, init_params=False)
data, label = AddInput(test_model, batch_size=100, db=os.path.join(data_folder, "mnist-test-nchw-lmdb"),
                      db_type='lmdb')
softmax = AddModel(test_model, data)
AddAccuracy(test_model, softmax, label)
deploy_model = model_helper.ModelHelper(name="mnist_deploy", arg_scope=arg_scope, init_params=False)
AddModel(deploy_model, "data")

workspace.RunNetOnce(train_model.param_init_net)
workspace.CreateNet(train_model.net, overwrite=True)
total_iters = 200
accuracy = np.zeros(total_iters)
loss = np.zeros(total_iters)
for i in range(total_iters):
    workspace.RunNet(train_model.net)
    accuracy[i] = workspace.blobs["accuracy"]
    loss[i] = workspace.blobs['loss']
    if i % 25 == 0:
        print("Iter: {}, loss: {}, accuracy: {}".format(i, loss[i], accuracy[i]))
plt.plot(loss, 'b')
plt.plot(accuracy, 'r')
plt.title("Summary of Training Run")
plt.xlabel("Iteration")
plt.legend(("Loss", "Accuracy"), loc="upper right")
plt.show()
pe_meta = pe.PredictorExportMeta(
        predict_net=deploy_model.net.Proto(),
        parameters=[str(b) for b in deploy_model.params],
        inputs=["data"],
        outputs=["softmax"],)
pe.save_to_db("minidb", os.path.join(root_folder, "mnist_model.minidb"), pe_meta)
print("Deploy model saved to:" + root_folder + "/mnist_model.minidb")

blob = workspace.FetchBlob("data")
plt.figure()
plt.title("Batch of Testing Data")
_ = visualize.NCHW.ShowMultiple(blob)

workspace.ResetWorkspace(root_folder)
print("The blobs in the workspace after reset: {}".format(workspace.Blobs()))
predict_net = pe.prepare_prediction_net(os.path.join(root_folder, "mnist_model.minidb"), "minidb")
print("The blobs in the workspace after loading the model: {}".format(workspace.Blobs()))
workspace.FeedBlob("data", blob)
workspace.RunNetOnce(predict_net)
softmax = workspace.FetchBlob("softmax")
print("Shape of softmax: {}".format(softmax.shape))
curr_pred, curr_conf = max(enumerate(softmax[0]), key=operator.itemgetter(1))
print("Prediction: {}".format(curr_pred))
print("Confidence: {}".format(curr_conf))
plt.figure()
plt.title("Prediction for the first image")
plt.ylabel("Confidence")
plt.xlabel("Label")
_ = plt.plot(softmax[0], 'ro')
plt.show()
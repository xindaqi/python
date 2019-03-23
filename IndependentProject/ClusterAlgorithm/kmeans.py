# python2.7
from numpy import *
import matplotlib.pyplot as plt
from itertools import cycle

# 源数据
data = open('data.txt')

# <class '_io.TextIOWrapper'>
print(type(data))


# <_io.TextIOWrapper name='data.txt' mode='r' encoding='UTF-8'>
print(data)
tempDatas = []
datas = []
# 源数据分割及类型转换
for line in data.readlines():
	curLine = line.strip().split('\t')
	 # <type 'list'>
	print("Type of tempDatas element: {}".format(type(curLine)))
	# ['1.65', '4.28']
	print("curlLine data: {}".format(curLine))
	类型转换
	curLines = map(float, curLine)
	datas.append(curLines)
	
	# datas.append(curLines)
print("Source datas: {}".format(datas))
print("Type of source datas: {}".format(type(datas)))
# datas = map(float, datas)
print("Source datas: {}".format(datas))
# 转成矩阵形式
datasMat = mat(datas)


# <class 'numpy.matrixlib.defmatrix.matrix'>
print(type(datasMat))
print(datasMat)

# [[ 1.65  4.28]
#  [-3.4   3.4 ]
#  [ 4.8  -1.15]]
print("datasMat[0]: {}".format(datasMat[([0,1,2],[0,0,0])[0]]))

# (20, 2)
print(shape(datasMat))

# 20
print(shape(datasMat)[0])

# 20
print(len(datas))
# 源数据维度
m = shape(datasMat)[0]
n = shape(datasMat)[1]
print("shape row: {}".format(m))
# 聚类结果寄存器,第一列为簇,即组别
# 第二列为各点与中心点的距离
clusterAssment = mat(zeros((m, n)))

# <class 'numpy.matrixlib.defmatrix.matrix'>
print(type(clusterAssment))

# (20, 2) 0 0
print(clusterAssment)
# 最小值与最大值坐标
minJ0 = min(datasMat[:, 0])
minJ1 = min(datasMat[:, 1])
# -0.3
print("Minimum column zero: {}".format(minJ0))
print("Type of column zero: {}".format(type(minJ0)))
# -0.4
print("Minimum column one: {}".format(minJ1))

maxJ0 = max(datasMat[:, 0])
maxJ1 = max(datasMat[:, 1])

# 4.8
print("Max column zero: {}".format(maxJ0))

# 4.28
print("Max column one: {}".format(maxJ1))

rangeJ0 = float(maxJ0 - minJ0)
rangeJ1 = float(maxJ1 - minJ1)
centroids = mat(zeros((4, 2)))
print("Center point: {}".format(centroids))
获取随机中心点
centroids[:, 0] = minJ0 + rangeJ0 * random.rand(4, 1)
centroids[:, 1] = minJ1 + rangeJ1 * random.rand(4, 1)
print(centroids)


# plt.show()

# <class 'numpy.matrixlib.defmatrix.matrix'>
print(type(clusterAssment))
# 迭代标志位
clusterChanged = True
while clusterChanged:
	clusterChanged = False
	for i in range(20):
		minDistance = inf 
		minIndex = -1
		for j in range(4):
			# 计算距离
			distance = sqrt(sum(power(centroids[j,:] - datasMat[i, :], 2)))
			if distance < minDistance:
				# 判断距离,并更新
				minDistance = distance
				# 记录分组
				minIndex = j
		# 判断聚类结果是否改变,改变则继续迭代,直至收敛
		if clusterAssment[i, 0] != minIndex:
			clusterChanged = True
		# 存储聚类结果,第一列为簇即组别,第二类为各点到中心点的距离
		clusterAssment[i, :] = minIndex, minDistance**2
		# print("Center of datas: {}".format(centroids))
		# print("Cluster assment: {}".format(clusterAssment))
	for center in range(4):
		pointsCluster = datasMat[nonzero(clusterAssment[:, 0].A == center)[0]]
		centroids[center, :] = mean(pointsCluster, axis=0)
	# 中心点结果
	print("Center of points: {}".format(centroids))
# 聚类结果
print("Cluster result: {}".format(clusterAssment))
# 获取组别标签,转为list
label = clusterAssment[:, 0].tolist()

print("Cluster label: {}".format(label))
# 按住组别绘制对应颜色的点
mark = ['or', 'ob', 'og', 'oy', '^r', '+r', 'sr', 'dr', '<r', 'pr']
# 按照组别绘制对应颜色的中心点
color = ['*r', '*b', '*g', '*y']
j = 0

for i in range(4):
	plt.plot(centroids[i,0], centroids[i, 1],color[i] ,markersize=12)
# 类别标签为list格式[[0.0],[1.0]]
# 提取数据并转化为int类型
for i in label:
	plt.ion()
	plt.plot(datasMat[j,0], datasMat[j, 1], mark[int(i[0])], markersize=5)
	plt.xlabel("X/cm")
	plt.ylabel("Y/cm")
	plt.title("Cluster Resutls Showing")
	j += 1

	print("Type of label: {}".format(type(i)))
	print("Label : {}".format(i[0]))
	# plt.scatter()
	plt.savefig("results/cluster.png",format='png')
plt.show()
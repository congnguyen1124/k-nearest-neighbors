from sklearn import datasets
import numpy as np
import math
import operator

def cal_discence(point1, point2):
    dimension = len(point1)
    distance = 0

    for i in range(dimension):
        distance += (point1[i] - point2[i])* (point1[i] - point2[i])

    return math.sqrt(distance)

def get_k_neighbors(training_X, label_y, point, k):
    distances = []
    # Calculate discence point to all element in X
    neighbors = []
    for i in range(len(training_X)):
        distance = cal_discence(training_X[i], point)
        distances.append((distance, label_y[i]))
    distances.sort(key=operator.itemgetter(0)) # sort by distance
    for i in range(k):
        neighbors.append(distances[i][1])
    return neighbors

def highest_votes(neighbors_labels):
    label_count = [0,0,0]
    for label in neighbors_labels:
        label_count[label] += 1
    max_count = max(label_count)
    return label_count.index(max_count)

def predict(training_X, label_y, point, k):
    neighbors_labels = get_k_neighbors(training_X, label_y, point, k)
    return highest_votes(neighbors_labels)

def accuracy_score(predict, tests):
    total = len(predict)
    correct_count = 0
    for i in range(total):
        if predict[i] == tests[i]:
            correct_count += 1

    accuracy = correct_count / total * 100

    return accuracy

iris = datasets.load_iris() 

# data (petal length, petal width, sepal length, sepal width)

iris_x = iris.data
iris_y = iris.target

rand_index = np.arange(iris_x.shape[0])
np.random.shuffle(rand_index)

iris_x = iris_x[rand_index]
iris_y = iris_y[rand_index]

X_train = iris_x[:100,:]
X_test = iris_x[100:,:]
y_train = iris_y[:100]
y_test = iris_y[100:]

y_predict = []
for p in X_test:
    label = predict(X_train, y_train, p, 5)
    y_predict.append(label)

print(y_predict)
# print(y_test)

acc = accuracy_score(y_predict, y_test)
print('Accuracy = ' + str(acc) + '%')

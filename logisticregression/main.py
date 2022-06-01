from sklearn.model_selection import train_test_split
from myutils import gradient_descent, normalize, sigmoid, loss

import numpy as np
import matplotlib.pyplot as plt


data = np.genfromtxt('logisticregression/diabetes.csv', delimiter=',')


xTrain, xTest, yTrain, yTest = train_test_split(normalize(data[:, :-1]),
                                                data[:, -1],
                                                test_size=0.2,
                                                random_state=1)

xTrain, xVal, yTrain, yVal = train_test_split(xTrain, yTrain,
                                              test_size=0.3, random_state=1)


def train(XTrain, YTrain, lr=0.001, maxIter=100):
    weights = np.zeros(XTrain.shape[1])

    history = []
    bias = 0

    for _ in range(maxIter):
        x_d_w = np.matmul(weights, XTrain.transpose()) + bias
        h = sigmoid(x_d_w)
        curr_loss = loss(YTrain, h)

        dw, db = gradient_descent(XTrain, YTrain, h)

        weights -= (lr * dw)
        bias -= (lr * db)

        history.append(curr_loss)

    return weights, bias, history


def predict(instance, weights, bias):
    x_d_w = np.matmul(instance, weights.transpose()) + bias
    h = sigmoid(x_d_w)
    return np.array([(1 if p > 0.5 else 0) for p in h])


lr_values = [0.1, 0.01, 0.001, 0.0001]
iter_values = [10, 50, 100, 500, 1000]

print()
print('VALIDATION')
print('-'*80)

for lr in lr_values:
    WEIGHTS, BIAS, HISTORY = train(xTrain, yTrain, lr, maxIter=500)
    predicted = predict(xVal, WEIGHTS, BIAS)

    print(f"VALIDATION ACCURACY FOR LR = {lr} =", (sum(
        predicted == yVal) / len(yVal))*100)

print('-'*80)

WEIGHTS, BIAS, HISTORY = train(xTrain, yTrain, 0.01, maxIter=500)
predicted = predict(xTest, WEIGHTS, BIAS)
print(f"TEST ACCURACY FOR LR = {0.01} =",
      (sum(predicted == yTest) / len(yTest))*100)

print()

average_tl = []

for i in iter_values:
    WEIGHTS, BIAS, HISTORY = train(xTrain, yTrain, 0.01, maxIter=i)
    predicted = predict(xVal, WEIGHTS, BIAS)
    print(f"VALIDATION ACCURACY FOR ITERS = {i} =", (sum(
        predicted == yVal) / len(yVal))*100)

    average_tl.append(np.mean(HISTORY))

print()

WEIGHTS, BIAS, HISTORY = train(xTrain, yTrain, 0.01, 1000)
predicted = predict(xTest, WEIGHTS, BIAS)
print(f"TEST ACCURACY FOR ITERS = {1000} =",
      (sum(predicted == yTest) / len(yTest))*100)


plt.plot(iter_values, average_tl, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Loss')
plt.show()

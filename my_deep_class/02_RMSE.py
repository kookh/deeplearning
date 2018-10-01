# 평균 제곱근 오차
# 오차를 줄여야 한다.

import numpy as np

# 기울기 a와 y절편 b
ab  = [3, 76]

data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data]
# print(x, type(x))
y = [i[1] for i in data]

def predict(x) :
    return ab[0]*x + ab[1]

# 평균 제곱근 오차
# p는 예측값, a는 실제값
def rmse(p, a) :
    return np.sqrt(((p-a) ** 2).mean())

# print (type(np.array(y)))

def rmse_val(predict_result, y):
    return rmse(np.array(predict_result), np.array(y))

predict_result= []
for i in range(len(x)):
    predict_result.append(predict(x[i]))
    print("공부한 시간=%.f, 실제 점수=%.f, 예측 점수=%.f" % (x[i], y[i], predict(x[i])))

# 오차
print("rmse 최종값: " + str(rmse_val(predict_result, y)))
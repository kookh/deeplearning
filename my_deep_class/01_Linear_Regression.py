# 선형 회귀 : 방정식
# 기울기 : 최소제곱법

import numpy as np

x = [2, 4, 6, 8]
y = [81, 93, 91, 97]

mx = np.mean(x)
my = np.mean(y)
print("x의 평균값:", mx)
print("y의 평균값:", my)
# print(mx, my)

# for i in x:
#     print(i)

# divisor : 분모
divisor = sum([(mx - i)**2 for i in x])
print("분모:", divisor)

# dividend : 분자
def top(x, mx, y, my):
    d = 0
    for i in range(len(x)):
        d += (x[i] - mx) * (y[i] - my)
    return d

dividend = top(x, mx, y, my)
print("분자:", dividend)


a = dividend / divisor
b = my - (mx*a)

print("기울기 a =", a)
print("y 절편 b =", b)
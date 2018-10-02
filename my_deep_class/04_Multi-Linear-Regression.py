# 경사하강법
# 기울기가 0인 부분을 찾는다.
# 최소 제곱법 공식을 사용하지 않고, 평균 제곱근 오차를 구하고, 경사 하강법을 기울기 a와 y절편 b값을 구할 수 있다.

import tensorflow as tf

data = [[2, 0, 81], [4, 4, 93], [6, 2, 91], [8, 3, 97]]
x1 = [x_row1[0] for x_row1 in data]
x2 = [x_row2[1] for x_row2 in data]
y_data = [y_row[2] for y_row in data]


# 기울기의 범위는 0 ~ 10 사이이며, y 절편은 0 ~ 100 사이에서 랜덤으로 설정한다.
a1 = tf.Variable(tf.random_uniform([1], 0, 10, dtype = tf.float64, seed = 0))
a2 = tf.Variable(tf.random_uniform([1], 0, 10, dtype = tf.float64, seed = 0))
b = tf.Variable(tf.random_uniform([1], 0, 100, dtype = tf.float64, seed = 0))

learning_rate = 0.1

y = (a1 * x1) + (a2 * x2) + b

# 평균 제곱근 오차 식
rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_data)))

gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(gradient_decent)
        if step % 100 == 0:
            print("Epoch: %.f, RMSE = %.04f, 기울기 a1 = %.4f, 기울기 a2 = %.4f, y 절편 b = %.4f" % (step, sess.run(rmse), sess.run(a1), sess.run(a2), sess.run(b)))



# 경사하강법
# 기울기가 0인 부분을 찾는다.
# 최소 제곱법 공식을 사용하지 않고, 평균 제곱근 오차를 구하고, 경사 하강법을 기울기 a와 y절편 b값을 구할 수 있다.

import tensorflow as tf

data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x_data = [x_row[0] for x_row in data]
y_data = [y_row[1] for y_row in data]

a = tf.Variable(tf.random_uniform([1], 0, 10, dtype = tf.float64, seed = 0))
b = tf.Variable(tf.random_uniform([1], 0, 100, dtype = tf.float64, seed = 0))

learning_rate = 0.1

y = a * x_data + b

# 평균 제곱근 오차 식
rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_data)))

gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(gradient_decent)
        if step % 100 == 0:
            print("Epoch: %.f, RMSE = %.04f, 기울기 a = %.4f, y 절편 b = %.4f" % (step, sess.run(rmse), sess.run(a), sess.run(b)))



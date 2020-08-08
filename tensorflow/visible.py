import tensorflow as tf


# 变量op
a = tf.constant([1,2,3,4,5])

var = tf.Variable(tf.random_normal([2,3], mean=0.0, stddev=1.0))

print(a, var)

# 存在变量时，一定要---->初始化op并且运行
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # 运行op
    sess.run(init_op)

    # 把程序图结构写入事件文件
    graph = tf.summary.FileWriter("./summary/test", graph=tf.get_default_graph())

    print(sess.run([a,var]))
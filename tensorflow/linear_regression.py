import tensorflow as tf
import os

def myregression():
    """
    自实现线性回归预测
    :return: None
    """
    #1、准备数据，x 特征值[100, 1] y 目标值[100]
    x = tf.random_normal([100,1], mean=0.0, stddev=1.0)
    y_true = tf.matmul(x, [[0.7]]) + 0.8

    #2、 建立线性回归模型 1个权重 1个偏执
    weight = tf.Variable(tf.random_normal([1,1], mean=0.0, stddev=1.0), name="w")
    bias = tf.Variable(0.0, name="b")

    y_predict=tf.matmul(x,weight) + bias

    #3、建立损失函数
    loss = tf.reduce_mean(tf.square(y_true - y_predict))

    #4、梯度下降
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    #定义一个初始化变量op
    init_op = tf.global_variables_initializer()

    # 收集变量
    tf.summary.scalar("loss", loss)
    tf.summary.histogram("weights", weight)

    # 定义合并tensor的op
    merged = tf.summary.merge_all()


    # 定义一个保存模型的实例
    saver = tf.train.Saver()

    #通过绘画运行程序
    with tf.Session() as sess:
        sess.run(init_op)

        #打印随机最先初始的权重和偏置
        print("随即初始化变量weight为：%f, bias为：%f" % (weight.eval(), bias.eval()))

        # 建立事件文件
        filewriter = tf.summary.FileWriter("./summary/test", graph=tf.get_default_graph())

        # 加载模型,注意路径是不一样的：检查有没有checkpoint，加载模型名字ckp
        if os.path.exists("./temp/checkpoint"):
            saver.restore(sess, "./temp/ckp")

        #循环优化
        for i in range(20):
            sess.run(train_op)

            # 运行合并的tensor
            summary = sess.run(merged)
            filewriter.add_summary(summary, i)

            print("第%d次变量weight为：%f, bias为：%f" % (i, weight.eval(), bias.eval()))

        saver.save(sess, "./temp/ckp")
    return None


if __name__ == "__main__":
    myregression()

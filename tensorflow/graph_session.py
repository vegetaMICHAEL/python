import tensorflow as tf
import os


# 创建一张图，上下文环境,两张图互不干扰
g = tf.Graph()
print(g)
with g.as_default():
    c = tf.constant(12.0)
    print(c.graph)

# 实现一个加法运算
a = tf.constant(5.0)
b = tf.constant(6.0)
sum = tf.add(a, b)
print(sum)

# 获取默认的图，分配一段内存
graph = tf.get_default_graph()
print(graph)

# placehoder是一个占位符,feed_dict是一个字典
plt = tf.placeholder(tf.float32, [2, 3])


# 会话只使用一张图，
# 1、可以指定一张图运行with tf.Session(graph=g) as sess:
# 2、可以查看运行在哪个设备with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess
"""
with context_expression [as target(s)]:
with-body

with：释放资源、关闭文件、释放线程
建立运行时上下文负责执行 with 语句块上下文中的进入与退出操作。通常使用 with 语句调用上下文管理器，也可以通过直接调用其方法来使用。
上下文表达式：with之后，要返回一个上下文管理器对象
__enter__()：语句体执行之前进入运行时上下文
__exit__() ：语句体执行完后从运行时上下文退出
"""
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run([a, b, sum]))
    print(sess.run(plt, feed_dict={plt: [[1,2,3],[4,5,6]]}))
    print(a.graph)
    print(b.graph)
    print(sum.graph)
    # 取值
    print(sum.eval())
    print(a.graph)
    print(a.op)
    print(a.shape)
    print(a.name)

# tensor打印的形状表示
# 0维：()  1维:(5)  2维:(5,6)  3维:(2,3,4)
# 形状：静态性状：tf.Tensor.get_shape():获取静态性状；tf.Tensor.set_shape():更新静态性状
#       动态形状：tf.reshape：创建一个具有不同动态形状的新张量
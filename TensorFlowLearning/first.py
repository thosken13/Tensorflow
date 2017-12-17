import tensorflow as tf

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) #also is tf.float32, implicitly
print(node1, node2) #doesn't print their values, because they are nodes which
                    #output their values when evaluated in a session

sess = tf.Session()
print(sess.run([node1, node2]))

node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))

#add operations for variables to the graph
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adderNode = a + b #shortcut for tf.add(a, b)
print(sess.run(adderNode, {a: 3, b: 4.5}))
print(sess.run(adderNode, {a: [1, 3], b: [2, 4]}))

addAndTriple = adderNode*3.0
print(sess.run(addAndTriple, {a: 3, b: 4.5}))

print("-----Building a model-----")
#trainable parameters
W = tf.Variable([3.0], dtype=tf.float32)
b = tf.Variable([-3.0], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linearModel = W*x + b
init = tf.global_variables_initializer() #variables are not initialised when declared
sess.run(init) #variables still not initialised until init is run
print(sess.run(linearModel, {x: [1,2,3,4]}))

#have an output so that we can find the error
y = tf.placeholder(tf.float32)
squaredDeltas = tf.square(linearModel - y)
loss = tf.reduce_sum(squaredDeltas)
print("Loss", sess.run(loss, {x: [1,2,3,4], y: [0,-1,-2,-3]}))

print("Fix Model variables to correct values")
#fix the model variables for the desired output
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

print("Train model using gradient descent")
#train to produce correct model variables
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init) #reset values to incorrect defaults
for i in range(1000):
    sess.run(train, {x: [1,2,3,4], y: [0,-1,-2,-3]})
print(sess.run([W, b]))















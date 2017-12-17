import numpy as np
import tensorflow as tf

def modelFunc(features, labels, mode):
    #linear model
    W = tf.get_variable("W", [1], dtype=tf.float64)
    b = tf.get_variable("b", [1], dtype=tf.float64)
    y = W*features["x"] + b
    #Loss sub-graph
    loss = tf.reduce_sum(tf.square(y - labels))
    # Training sub-graph
    globalStep = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(globalStep, 1))
    #EstimatorSpec connects subgraphs we built to the appropriate functionality
    return tf.estimator.EstimatorSpec(mode=mode, predictions=y, loss=loss, train_op=train)

estimator = tf.estimator.Estimator(model_fn = modelFunc)

xTrain = np.array([1., 2., 3., 4.])
yTrain = np.array([0., -1., -2., -3.])
xEval = np.array([2., 5., 8., 1.])
yEval = np.array([-1.01, -4.1, -7, 0.])
inputFunc = tf.estimator.inputs.numpy_input_fn({"x": xTrain}, yTrain, batch_size=4, num_epochs=None, shuffle=True)
trainInputFunc = tf.estimator.inputs.numpy_input_fn({"x": xTrain}, yTrain, batch_size=4, num_epochs=1000, shuffle=False)
evalInputFunc = tf.estimator.inputs.numpy_input_fn({"x": xEval}, yEval, batch_size=4, num_epochs=1000, shuffle=False)
#invoke training with 1000 steps
estimator.train(inputFunc, steps=1000)

#evaluate its performance
trainMetrics = estimator.evaluate(input_fn=trainInputFunc)
evalMetrics = estimator.evaluate(input_fn=evalInputFunc)

print("train metrics: {}".format(trainMetrics))
print("eval metrics: {}".format(evalMetrics))
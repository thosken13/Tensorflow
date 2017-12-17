import numpy as np
import tensorflow as tf

#declare list of features (we have only one numeric feature)
featureColumns = [tf.feature_column.numeric_column("x", shape=[1])]
#estimator is front for training and evaluation
estimator = tf.estimator.LinearRegressor(feature_columns=featureColumns)

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
# We run experiments to analyze the potential advantages of updating the architecture of a Deep Neural Network during
# optimization, and simultaneously using as solution an ensemble of weighted networks (see below).  It is crucial to
# observe that during training we obtain:

# -The optimum parametrization of the network (as in the classical sense).
# -The optimal architecture of the network, which is variable during runtime.
# -A set of "ensemble" weights that minimizes the objective function.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import trainAndEvaluate
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.boston_housing.load_data())

# @title AdaNet parameters
LEARNING_RATE = 0.001   # @param {type:"number"}
TRAIN_STEPS = 1000      # @param {type:"integer"} 100000
BATCH_SIZE = 32         # @param {type:"integer"}


# Use variable weights that appear in linear combination of subnetworks.
# LEARN_MIXTURE_WEIGHTS # @param {type:"boolean"}

# lambda parameter serves to prevent the optimization from assigning too
# much weight to more complex subnetworks
# ADANET_LAMBDA         # @param {type:"number"}

# Size of adaptive architecture of AdamNet.
# BOOSTING_ITERATIONS = 5  # @param {type:"integer"}

loss_results = []
for i in range(1, 4):
    results, _ = trainAndEvaluate.train_and_evaluate(x_train, y_train, x_test, y_test,
                                                     LEARNING_RATE,
                                                     TRAIN_STEPS,
                                                     BATCH_SIZE,
                                                     learn_mixture_weights=False,
                                                     adanet_lambda=0,
                                                     boosting_iterations=i)
    loss_results.append(results["average_loss"])
    print(loss_results)
    print("Loss:", results["average_loss"])
    print("Architecture:", trainAndEvaluate.ensemble_architecture(results))

plt.scatter(list(range(1, 4)), loss_results)
plt.show()

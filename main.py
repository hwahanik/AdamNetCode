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

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.boston_housing.load_data())

# @title AdaNet parameters
LEARNING_RATE = 0.001   # @param {type:"number"}
TRAIN_STEPS = 100000    # @param {type:"integer"}
BATCH_SIZE = 32         # @param {type:"integer"}

LEARN_MIXTURE_WEIGHTS = False   # @param {type:"boolean"}
ADANET_LAMBDA = 0               # @param {type:"number"}


results, _ = trainAndEvaluate.train_and_evaluate(x_train, y_train, x_test, y_test,
                                                 LEARNING_RATE,
                                                 TRAIN_STEPS,
                                                 BATCH_SIZE,
                                                 LEARN_MIXTURE_WEIGHTS,
                                                 ADANET_LAMBDA)

print("Loss:", results["average_loss"])
print("Architecture:", trainAndEvaluate.ensemble_architecture(results))


# We run experiments to analyze the potential advantages of updating the architecture of a Deep Neural Network during
# optimization, and simultaneously using as solution an ensemble of weighted networks (see below).  It is crucial to
# observe that during training we obtain:

# -The optimum parametrization of the network (as in the classical sense).
# -The optimal architecture of the network, which is variable during runtime.
# -A set of "ensemble" weights that minimizes the objective function.


# Variables that define the adaptive architecture
#
# LEARN_MIXTURE_WEIGHTS # @param {type:"boolean"}
# Use variable weights that appear in linear combination of subnetworks, for
# optimization of the mixture of subnetworks.

# ADANET_LAMBDA         # @param {type:"number"}
# lambda parameter serves to prevent the optimization from assigning too
# much weight to more complex subnetworks

# BOOSTING_ITERATIONS   # @param {type:"integer"}
# Size of adaptive architecture of AdaNet.

import matplotlib.pyplot as plt
import tensorflow as tf
import trainAndEvaluate
import numpy as np

# @title AdaNet parameters
LEARNING_RATE = 0.001       # @param {type:"number"}
TRAIN_STEPS = 10000         # @param {type:"integer"} 100000
BATCH_SIZE = 32             # @param {type:"integer"}
BOOSTING_ITERATIONS = 5     # @param {type:"integer"} Boosting iterations ==> Network depth.

# Testing the AdaNet architecture for the Boston Housing Dataset.
# 1. Load data
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.boston_housing.load_data())

# 2. Training and evaluating the model.

# 2. a. -Train with no optimization of the mixture weights.
#       -Lambda=0;
#       -Increasing size of the different networks.
#       -Constant learning rate, and constant batch size.

# Experiments


def training_results(x_tr, y_tr, x_tst, y_tst, a_lambda, learn_mixtures):
    results, _ = trainAndEvaluate.train_and_evaluate(x_tr, y_tr, x_tst, y_tst,
                                                     LEARNING_RATE,
                                                     TRAIN_STEPS,
                                                     BATCH_SIZE,
                                                     learn_mixtures,
                                                     a_lambda,
                                                     boosting_iterations=BOOSTING_ITERATIONS)
    print("Architecture:", trainAndEvaluate.ensemble_architecture(results))
    print(results["average_loss"])
    print("Uniform average loss:", results["average_loss/adanet/uniform_average_ensemble"])
    print("Adanet lambda", a_lambda)
    return [results["average_loss"], results["average_loss/adanet/uniform_average_ensemble"]]


# Training without learning mixtures of deep networks, learn_mixtures = false
# lambda = [0.1, 0.15, 0.2, 0.25 ... 1]

loss_results = []
uniform_loss_results = []
adanet_lambda = np.linspace(0, 0.2, 11)
learnMixtures = True

# Training the ensemble, and calculating losses
for l in adanet_lambda:
    loss = training_results(x_train, y_train, x_test, y_test, l, learnMixtures)[0]
    uniform_loss = training_results(x_train, y_train, x_test, y_test, l, learnMixtures)[1]
    loss = float("{0:.2f}".format(loss))
    uniform_loss_results.append(uniform_loss)
    loss_results.append(loss)
#

print(adanet_lambda)
print(loss_results)
print(uniform_loss_results)


# Plotting results

plt.ioff()
epsilon = 1e-2

fig, ax = plt.subplots()
ax.scatter(adanet_lambda, loss_results, color="blue", alpha=0.5)
ax.scatter(adanet_lambda, uniform_loss_results, c="green", alpha=0.5, )

for i, txt in enumerate(loss_results):
    ax.annotate(txt, (adanet_lambda[i], loss_results[i]))

title = "AdaNet ensemble: uniform (black) vs regularized (red), for depth = 5"
ax.plot(adanet_lambda, loss_results, color="red")

ax.plot(adanet_lambda, uniform_loss_results, color="black")
l_range = np.linspace(0, 1, 6)

plt.title(title, color="b", size="medium")
plt.xlabel("regularization parameter", color="b", size="medium")
plt.ylabel("MSE (loss)", color="b", size="medium")
plt.xticks(l_range)

plt.ylim([0.045, 0.065])
plt.draw()
plt.show()

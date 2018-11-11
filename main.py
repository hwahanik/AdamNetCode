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

# @title AdaNet parameters
LEARNING_RATE = 0.001   # @param {type:"number"}
TRAIN_STEPS = 10000     # @param {type:"integer"} 100000
BATCH_SIZE = 32         # @param {type:"integer"}
SIZE_ADAPTIVE = 6       # @param {type:"integer"} Network depth allowed to adapt.

# Testing the AdaNet architecture for the Boston Housing Dataset.
# 1. Load data
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.boston_housing.load_data())


# 2. Training and evaluating the model.

# 2. a. -Train with no optimization of the mixture weights.
#       -Lambda=0;
#       -Increasing size of the different networks.
#       -Constant learning rate, and constant batch size.

loss_results = []
architectures = list(range(1, SIZE_ADAPTIVE+1))
for i in architectures:
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

plt.ioff()
fig, ax = plt.subplots()
ax.scatter(architectures, loss_results, c="b", alpha=0.5)
x_int_range = range(min(architectures), max(architectures)+1)
for i, txt in enumerate(loss_results):
    ax.annotate(txt, [architectures[i], loss_results[i]])
plt.title("AdamNet training")
plt.xlabel("Network depth")
plt.ylabel("MSE (loss)")
plt.xticks(x_int_range)
plt.ylim([0, max(loss_results)])
plt.draw()
plt.show()

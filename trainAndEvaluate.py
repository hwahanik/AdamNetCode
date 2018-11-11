# Train and evaluate (Boston Housing Market Example)

# Here we create an adanet.Estimator using the SimpleDNNGenerator.
# It is possible to study the the effects of two hyperparams: learning mixture
# weights and complexity regularization.
#
# At first we will not learn the mixture weights, using their default initial value. Here
# they will be scalars initialized to $1/N$ where $N$ is the number of subnetworks in the
# ensemble, effectively creating a uniform average ensemble.

import adanet
import tensorflow as tf
import subnetworkGenerator
import utils


def train_and_evaluate(x_train, y_train, x_test, y_test,
                       learning_rate,
                       train_steps,
                       batch_size,
                       learn_mixture_weights,
                       adanet_lambda, boosting_iterations):
    """Trains an adanet.Estimator` to predict housing prices."""

    estimator = adanet.Estimator(
      # Since we are predicting housing prices, we'll use a regression
      # head that optimizes for MSE.
      head=tf.contrib.estimator.regression_head(
          loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE),

      # Define the generator, which defines our search space of subnetworks
      # to train as candidates to add to the final AdaNet model.
      subnetwork_generator=subnetworkGenerator.SimpleDNNGenerator(
          optimizer=tf.train.RMSPropOptimizer(learning_rate),
          learn_mixture_weights=learn_mixture_weights,
          seed=utils.RANDOM_SEED),

      # Lambda is a the strength of complexity regularization. A larger
      # value will penalize more complex subnetworks.
      adanet_lambda=adanet_lambda,

      # The number of train steps per iteration.
      max_iteration_steps=train_steps // boosting_iterations,

      # The evaluator will evaluate the model on the full training set to
      # compute the overall AdaNet loss (train loss + complexity
      # regularization) to select the best candidate to include in the
      # final AdaNet model.
      evaluator=adanet.Evaluator(
          input_fn=utils.input_fn(x_train, y_train, x_test, y_test,
                                  "train", training=False, batch_size=batch_size)),

      # The report materializer will evaluate the subnetworks' metrics
      # using the full training set to generate the reports that the generator
      # can use in the next iteration to modify its search space.
      report_materializer=adanet.ReportMaterializer(
          input_fn=utils.input_fn(x_train, y_train, x_test, y_test, "train", training=False, batch_size=batch_size)),

      # Configuration for Estimators.
      config=tf.estimator.RunConfig(
          save_checkpoints_steps=50000,
          save_summary_steps=50000,
          tf_random_seed=utils.RANDOM_SEED))

    # Train and evaluate using the tf.estimator tooling.
    train_spec = tf.estimator.TrainSpec(
        input_fn=utils.input_fn(x_train, y_train, x_test, y_test, "train", training=True, batch_size=batch_size),
        max_steps=train_steps)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=utils.input_fn(x_train, y_train, x_test, y_test, "test", training=False, batch_size=batch_size),
        steps=None)

    return tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def ensemble_architecture(result):
    """Extracts the ensemble architecture from evaluation results."""

    architecture = result["architecture/adanet/ensembles"]
    # The architecture is a serialized Summary proto for TensorBoard.
    summary_proto = tf.summary.Summary.FromString(architecture)
    return summary_proto.value[0].tensor.string_val[0]
import matplotlib.pyplot as plt
from termcolor import colored
import tensorflow as tf

plt.interactive(True)

FEATURES_KEY = "x"
# The random seed to use.
RANDOM_SEED = 42


def show_shapes(x_train, y_train, x_test, y_test, color='green'):
    print(colored ('Training shape:', color, attrs=['bold']))
    print('  x_train.shape:', x_train.shape)
    print('  y_train.shape:', y_train.shape)
    print(colored ('\nTesting shape:', color, attrs=['bold']))
    print('  x_test.shape:', x_test.shape)
    print('  y_test.shape:', y_test.shape)


def plot_data(my_data, cmap=None):
    plt.axis('off')
    fig = plt.imshow(my_data, cmap=cmap)
    plt.show(block=True)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    print(fig)


def show_sample(x_train, y_train, idx=0, color='blue'):
    print(colored('x_train sample:', color, attrs=['bold']))
    print(x_train[idx])
    print(colored('\ny_train sample:', color, attrs=['bold']))
    print(y_train[idx])


def show_sample_image(x_train, y_train, idx=0, color='blue', cmap=None):
    print(colored('Label:', color, attrs=['bold']), y_train[idx])
    print(colored('Shape:', color, attrs=['bold']), x_train[idx].shape)
    print()
    plot_data(x_train[idx], cmap=cmap)


# For supplying the data in TensorFlow. Using the tf.estimator.Estimator convention, define
# a function that returns an input_fn which returns feature and label Tensors.

def input_fn(x_train, y_train, x_test, y_test, partition, training, batch_size):
    """Generate an input function for the Estimator."""

    def _input_fn():

        if partition == "train":
            dataset = tf.data.Dataset.from_tensor_slices(({
                FEATURES_KEY: tf.log1p(x_train)
            }, tf.log1p(y_train)))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(({
                FEATURES_KEY: tf.log1p(x_test)
            }, tf.log1p(y_test)))

        # We call repeat after shuffling, rather than before, to prevent separate
        # epochs from blending together.
        if training:
            dataset = dataset.shuffle(10 * batch_size, seed=RANDOM_SEED).repeat()

        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

    return _input_fn
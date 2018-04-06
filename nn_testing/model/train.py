from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import basic libraries.
import numpy as numpy
import tensorflow as tf
import caltech101_images as images

# Set logging functionality.
tf.logging.set_verbosity(tf.logging.INFO)



# Drive Tensorflow application if main.
if __name__ == "__main__":
    tf.app.run();
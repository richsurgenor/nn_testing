from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import local packages.
import sys
from click.decorators import argument
from tensorflow.contrib.timeseries.examples import predict
from sys import argv
from PIL import Image
import numpy
sys.path.append('../')

# Imports
import numpy as np
import tensorflow as tf
import home_damage_images.home_images as home_images
import global_variables


tf.logging.set_verbosity(tf.logging.INFO)

IMAGE_RESIZE_WIDTH = 200
IMAGE_RESIZE_LENGTH = 200

POOL_SIZE_X = 2
POOL_SIZE_Y = 2

def cnn_model_fn(features, labels, mode):
    
    print(features)
    print(labels)
    
    # Input Layer
    input_layer = features

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=(64),
        kernel_size=[3,3],
        padding="same",
        activation=tf.nn.relu)

    conv3 = tf.layers.conv2d(
        inputs = conv1,
        filters = (64),
        kernel_size=[3,3],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[POOL_SIZE_X, POOL_SIZE_Y], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=(256),
        kernel_size=[3,3],
        padding="same",
        activation=tf.nn.relu)
    
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[POOL_SIZE_X, POOL_SIZE_Y], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, int(50 * 50 * 256)])

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=5)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1, name="classes_tensor"),

        # Add 'softmax_tensor' to the graph. It is used for PREDICT and by the 'logging_hook'
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
        
        "logits_abs": tf.abs(logits, name="log_abs"),
        
        "labels": tf.argmax(input=labels, axis=1, name="labels_tensor")
    }

    # print(predictions)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    #print(labels)
    #print(logits)

    #sess = tf.InteractiveSession()
    #with sess.as_default():
        #print("Labels: \n")
        #print(labels.eval())
        #print("\nLogits: \n")
        #print(logits.eval())

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
        
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, 
                                          train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=predictions["labels"], predictions=predictions["classes"])}

    print(eval_metric_ops)

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):

    # Get command line arguments.
    argument_dict = {}
    arguments_raw = argv
    
    # Parses command line arguments.
    while arguments_raw:
        if (arguments_raw[0][0] == "-"):
            argument_dict[arguments_raw[0]] = arguments_raw[1]
        arguments_raw = arguments_raw[1:] 
        
    #print(argument_dict)
    # Load training and eval data
    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    
    #train_data = mnist.train.images
    #train_data = home_images.train_set

    #train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    #eval_data = mnist.test.images
    #eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    home_images_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=global_variables.MODEL_SAVE_DIRECTORY)

    #tensors_to_log = {"probabilities": "softmax_tensor", "labels": "labels_tensor", "classes": "classes_tensor"}
    tensors_to_log = {}#"labels": "labels_tensor", "classes": "classes_tensor", 
                      #"probs" : "softmax_tensor", "logits" : "log_abs"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)

    if "evaluate" not in argv and "-predict" not in argument_dict:

        # Incrementally save
        try:
            while True:
                home_images_classifier.train(
                    input_fn=home_images.train_set,
                    hooks = [logging_hook],
                    steps=30)
    
        except KeyboardInterrupt:
            pass

    # Evaluate the model and print results
    #eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #    x={"x": home_images.eval_set},
    #    y=eval_labels,
    #    num_epochs=1,
    #    shuffle=False)

    
    eval_results = home_images_classifier.evaluate(input_fn=home_images.eval_set)
    #train_eval_results = home_images_classifier.evaluate(input_fn=home_images.train_set)
        
    if "-predict" in argument_dict:
        prediction_image = tf.read_file(argument_dict["-predict"])
        prediction_image = tf.image.decode_image(prediction_image, channels=3)
        prediction_image = tf.image.resize_images(prediction_image, [IMAGE_RESIZE_WIDTH, IMAGE_RESIZE_LENGTH])
        
        prediction_image_data = tf.data.Dataset.from_tensor_slices(prediction_image)
        
        print(home_images_classifier.predict(prediction_image_data))
        
    print(eval_results)
    #print(train_eval_results)

if __name__ == "__main__":    
    tf.app.run()    
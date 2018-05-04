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
import home_global_variables


tf.logging.set_verbosity(tf.logging.INFO)

IMAGE_RESIZE_WIDTH = 200
IMAGE_RESIZE_LENGTH = 200

POOL_SIZE_X = 2
POOL_SIZE_Y = 2 
                         

def cnn_model_fn(features, labels, mode):
    
    # Tensor holding possible category numbers (0 thru 4)
    cat_tensor = tf.constant(value=[0,1,2,3,4], 
                             dtype=tf.float32)
    
    print(features)
    print(labels)
    
    # Input Layer w/ dropout.
    input_layer = tf.layers.dropout(inputs=features,
                                    rate=0.2,
                                    training=mode==tf.estimator.ModeKeys.TRAIN)

    # Training parameter for the BN layers.
    BN_mode = True

    normal_conv1 = tf.layers.conv2d(inputs=input_layer,
                                    filters=32,
                                    strides=2,
                                    kernel_size=[3,3],
                                    activation=tf.nn.relu,
                                    padding='SAME')

    # First layer before pooling
    sep_conv1 = tf.layers.separable_conv2d(inputs=normal_conv1,
                                           filters=64,
                                           activation=tf.nn.relu,
                                           kernel_size=[3,3],
                                           padding='SAME')
    
    batch_norm1 = tf.nn.relu(tf.layers.batch_normalization(inputs=sep_conv1,
                                                           training=BN_mode,
                                                           name="batch_norm_1"))

    # 2nd layer before pooling
    sep_conv1_1 = tf.layers.separable_conv2d(inputs=batch_norm1,
                                           filters=128,
                                           activation=tf.nn.relu,
                                           strides=(2,2),
                                           kernel_size=[3,3],
                                           padding='SAME')
                                           
    
    batch_norm1_1 = tf.nn.relu(tf.layers.batch_normalization(inputs=sep_conv1_1,
                                                  training=BN_mode))

    # 3rd layer before pooling
    sep_conv1_2 = tf.layers.separable_conv2d(inputs=batch_norm1_1,
                                           filters=128,
                                           activation=tf.nn.relu,
                                           kernel_size=[3,3],
                                           padding='SAME')
    
    batch_norm1_2 = tf.nn.relu(tf.layers.batch_normalization(inputs=sep_conv1_2,
                                                             training=BN_mode))

    # 4th layer before pooling
    sep_conv1_3 = tf.layers.separable_conv2d(inputs=batch_norm1_2,
                                           filters=256,
                                           activation=tf.nn.relu,
                                           strides=(2,2),
                                           kernel_size=[3,3],
                                           padding='SAME')
    
    batch_norm1_3 = tf.nn.relu(tf.layers.batch_normalization(inputs=sep_conv1_3,
                                                             training=BN_mode))

    # 5th layer before pooling
    sep_conv1_4 = tf.layers.separable_conv2d(inputs=batch_norm1_3,
                                           filters=256,
                                           activation=tf.nn.relu,
                                           kernel_size=[3,3],
                                           padding='SAME')
    
    batch_norm1_4 = tf.nn.relu(tf.layers.batch_normalization(inputs=sep_conv1_4,
                                                             training=BN_mode))

    # 6th layer before pooling
    sep_conv1_5 = tf.layers.separable_conv2d(inputs=batch_norm1_4,
                                           filters=512,
                                           activation=tf.nn.relu,
                                           strides=(2,2),
                                           kernel_size=[3,3],
                                           padding='SAME')
    
    batch_norm1_5 = tf.nn.relu(tf.layers.batch_normalization(inputs=sep_conv1_5,
                                                             training=BN_mode))


    # 7th layer before pooling
    sep_conv1_6 = tf.layers.separable_conv2d(inputs=batch_norm1_5,
                                           filters=512,
                                           activation=tf.nn.relu,
                                           kernel_size=[3,3],
                                           padding='SAME')
    
    batch_norm1_6 = tf.nn.relu(tf.layers.batch_normalization(inputs=sep_conv1_6,
                                                             training=BN_mode))

    sep_conv1_7 = tf.layers.separable_conv2d(inputs=batch_norm1_6,
                                           filters=512,
                                           activation=tf.nn.relu,
                                           kernel_size=[3,3],
                                           padding='SAME')
    
    batch_norm1_7 = tf.nn.relu(tf.layers.batch_normalization(inputs=sep_conv1_7,
                                                             training=BN_mode))
    
    sep_conv1_8 = tf.layers.separable_conv2d(inputs=batch_norm1_7,
                                           filters=512,
                                           activation=tf.nn.relu,
                                           kernel_size=[3,3],
                                           padding='SAME')
    
    batch_norm1_8 = tf.nn.relu(tf.layers.batch_normalization(inputs=sep_conv1_8,
                                                             training=BN_mode))
    
    sep_conv1_9 = tf.layers.separable_conv2d(inputs=batch_norm1_8,
                                           filters=512,
                                           activation=tf.nn.relu,
                                           kernel_size=[3,3],
                                           padding='SAME')
    
    batch_norm1_9 = tf.nn.relu(tf.layers.batch_normalization(inputs=sep_conv1_9,
                                                             training=BN_mode))
    
    sep_conv1_10 = tf.layers.separable_conv2d(inputs=batch_norm1_9,
                                           filters=512,
                                           activation=tf.nn.relu,
                                           kernel_size=[3,3],
                                           padding='SAME')
    
    batch_norm1_10 = tf.nn.relu(tf.layers.batch_normalization(inputs=sep_conv1_10,
                                                             training=BN_mode))
    
    sep_conv1_11 = tf.layers.separable_conv2d(inputs=batch_norm1_10,
                                           filters=512,
                                           activation=tf.nn.relu,
                                           kernel_size=[3,3],
                                           padding='SAME')
    
    batch_norm1_11 = tf.nn.relu(tf.layers.batch_normalization(inputs=sep_conv1_11,
                                                             training=BN_mode))

    sep_conv1_12 = tf.layers.separable_conv2d(inputs=batch_norm1_11,
                                           filters=1024,
                                           activation=tf.nn.relu,
                                           strides=(2,2),
                                           kernel_size=[3,3],
                                           padding='SAME')
    
    batch_norm1_12 = tf.nn.relu(tf.layers.batch_normalization(inputs=sep_conv1_12,
                                                             training=BN_mode))
    
    sep_conv1_13 = tf.layers.separable_conv2d(inputs=batch_norm1_12,
                                           filters=1024,
                                           activation=tf.nn.relu,
                                           kernel_size=[3,3],
                                           padding='SAME')
    
    batch_norm1_13 = tf.nn.relu(tf.layers.batch_normalization(inputs=sep_conv1_13,
                                                             training=BN_mode))

    # First pool
    pool_1 = tf.layers.average_pooling2d(batch_norm1_13, 
                                         pool_size=[7,7], 
                                         strides=7)

    # Dense Layer
    pool10_flat = tf.reshape(pool_1, [-1, int(1 * 1 * 1024)])

    dense = tf.layers.dense(inputs=pool10_flat, units=1024)

    dropout = tf.layers.dropout(
        inputs=dense, rate=0.5, training=mode==tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=5)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1, name="classes_tensor"),

        # Add 'softmax_tensor' to the graph. It is used for PREDICT and by the 'logging_hook'
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
        
        "logits_abs": tf.identity(logits, name="log_abs"),
        
        "labels": tf.argmax(input=labels, axis=1, name="labels_tensor"),
        
        "mean": tf.reduce_sum(tf.multiply(cat_tensor, tf.nn.softmax(logits)), 
                              name="mean")
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

    #loss = loss + 0.1 * tf.losses.mean_pairwise_squared_error(labels=labels, 
    #                                        predictions=tf.nn.softmax(logits))

    #a = tf.argmax(logits, axis=1)
    #b = tf.argmax(labels, axis=1)
    
    #c = (a - b)**2

    #loss = tf.reduce_sum(tf.multiply(tf.cast(c, dtype=tf.float32), loss))

    #loss = loss + 0.1 * tf.losses.mean_squared_error(labels=tf.nn.softmax(labels),
    #                                    predictions=tf.nn.softmax(logits))

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        
        
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
    
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, 
                                          train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=predictions["labels"], predictions=predictions["classes"])}

    print(predictions["labels"])
    print(predictions["classes"])

    #init = tf.global_variables_initializer()
    #sess = tf.InteractiveSession()
    #sess.run(init)
    #with sess.as_default():
    #    print("Correct answers: " + str(predictions["labels"].eval()))    
    #    print("Predictions: " + str(predictions["classes"].eval()))

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
        model_fn=cnn_model_fn, model_dir=home_global_variables.MODEL_SAVE_DIRECTORY)

    #tensors_to_log = {"probabilities": "softmax_tensor", "labels": "labels_tensor", "classes": "classes_tensor"}
    tensors_to_log = {"labels": "labels_tensor", "classes": "classes_tensor", 
                      "probs" : "softmax_tensor",
                      "mean": "mean"}#, 
                      #"logits" : "log_abs"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1)

    if "evaluate" not in argv and "-predict" not in argument_dict:

        # Incrementally save
        try:
            while True:
                home_images_classifier.train(
                    input_fn=home_images.train_set,
                    hooks = [logging_hook],
                    steps=24)
    
        except KeyboardInterrupt:
            pass

    # Evaluate the model and print results
    #eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #    x={"x": home_images.eval_set},
    #    y=eval_labels,
    #    num_epochs=1,
    #    shuffle=False)

    
    eval_results = home_images_classifier.evaluate(input_fn=home_images.eval_set,
                                                   hooks=[logging_hook])
    
    
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
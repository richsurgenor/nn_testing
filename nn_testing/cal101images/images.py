import sys
sys.path.append('../')

import os
import tensorflow as tf
import global_variables

# Image dimension constants.
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 200

# Constant for eval/testing images split in percentage.
PERCENT_EVAL_IMAGES = 0.1

# Mini-batch size.
BATCH_SIZE = 32

# Number of categories to train on.
NUM_TRAINING_CATEGORIES = 10

label_dict = {}


# Input:  Directory name whose folders are labels that contain images of the 
#         given label.
#
# Output: Dictionary of paths to images and their labels. (keyed by path)
def gather_images(directory_name, category_count=0):
    eval_dictionary_out = {}
    train_dictionary_out = {}

    # Iterate through each folder, saving each file in each folder to a 
    # dictionary key.
    loop_index = 0
    category_index = 0
    for directory, paths, filenames in os.walk(directory_name):

        # Parse name of category out of directory name.
        category = directory[directory.rfind("/") + 1:]

        # Save category to name of labels
        label_dict[category] = category_index - 1

        # Must skip first iteration.
        if (category_index != 0):

            # Iterate and store each image in proper dictionary.
            for filename in filenames:
                # Choose if image should be put into train or eval dict.
                if (loop_index % (1/PERCENT_EVAL_IMAGES) == 0):
                    eval_dictionary_out[directory + "/" + filename] = category    
                else:
                    train_dictionary_out[directory + "/" + filename] = category    

                loop_index += 1

            # Must skip first iteration.
            if (category_index == category_count):
                break

            category_index += 1

        else:
            category_index += 1

    for key in label_dict:
        print("Key: %s  \t\t\t||  Label: %s" % (key, label_dict[key]))

    return train_dictionary_out, eval_dictionary_out


# Special case if calling programs want to call parse_images with the 
# global constant.
#
# Returns image directory, label dictionary
def gather_101_images(num_categories=0):
    return gather_images(global_variables.IMAGES_DIRECTORY, num_categories)


# Used for mapping file paths to actual images.
def _map_fn(image_path, label):
    NUM_CATEGORIES = 101

    # One hot encoding.
    one_hot = tf.cast(tf.one_hot(label, NUM_CATEGORIES), dtype=tf.int32)
    
    #sess = tf.InteractiveSession()
    #with sess.as_default():
    #    print(one_hot.eval())
    
    image_file = tf.read_file(image_path)
    image_decoded = tf.image.decode_image(image_file, channels=3)

    
    image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, IMAGE_HEIGHT, IMAGE_WIDTH)
    image_reformat = tf.image.convert_image_dtype(image_resized, dtype=tf.float32)
    
    image_resized = tf.reshape(image_reformat, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])

    return image_resized, one_hot

'''
# Used for mapping file paths to actual images.
def _map_fn_debug(image_path, label):
    sess = tf.InteractiveSession()
    
    NUM_CATEGORIES = 101

    # One hot encoding.
    one_hot = tf.one_hot(label, NUM_CATEGORIES)
    
    image_file = tf.read_file(image_path)
    image_decoded = tf.image.decode_image(image_file, channels=3)

    with sess.as_default():
        print("default: ")
        print(image_decoded.eval())

    image_decoded = tf.image.decode_image(image_file, channels=1)

    with sess.as_default():
        print("grayscale: ")
        print(image_decoded.eval())

    image_reformat = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
    image_resized = tf.image.resize_image_with_crop_or_pad(image_reformat, IMAGE_WIDTH, IMAGE_HEIGHT)
    #image_reshaped = tf.reshape(image_resized, [IMAGE_WIDTH, IMAGE_HEIGHT, 1])

    return image_resized, one_hot
'''

# Returns a training dataset and an evaluation dataset - in that order.
def create_training_set(image_label_dict):
    train_images = []
    train_labels = []

    # Parse the train images.
    for key in image_label_dict:
        # Append image path to train_images, and append its category to train_labels.
        train_images.append(key)
        train_labels.append(label_dict[image_label_dict[key]])
    print(train_labels)

    # Make TF dataset.
    train_images_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))

    #sess = tf.InteractiveSession()
    
   # with sess.as_default():
        #print(tf.constant(train_images).eval())
        #print(tf.constant(train_labels).eval())

    # Map dataset to (image data, one-hot) tuple.
    train_images_dataset = train_images_dataset.map(_map_fn)

    return train_images_dataset

def create_eval_set(image_label_dict):
    eval_images = []
    eval_labels = []

    # Parse the eval images.
    for key in image_label_dict:
        eval_images.append(key)
        eval_labels.append(label_dict[image_label_dict[key]])
    print(eval_labels)

    #print(eval_images)

    # Make TF dataset.
    eval_images_dataset = tf.data.Dataset.from_tensor_slices((eval_images, eval_labels))

    eval_images_dataset = eval_images_dataset.map(_map_fn)

    #sess = tf.InteractiveSession()
    
    #with sess.as_default():
        #print(tf.constant(eval_images).eval())
        #print(tf.constant(eval_labels).eval())

    return eval_images_dataset


# Returns a Tensorflow dataset for the 101 images.
def create_101_data_set(num_categories=0):
    images = gather_101_images(num_categories)

    print(len(images[0]))
    print(len(images[1]))

    training_data_set = create_training_set(images[0])
    evaluation_data_set = create_eval_set(images[1])

    return training_data_set, evaluation_data_set 


# For debugging -- this script is made to be called by other modules.
if __name__ == "__main__":
    #dict = create_101_data_set()
    gather_101_images()
    print(label_dict)

# Set eval and testing sets on import.
training_set = create_101_data_set(NUM_TRAINING_CATEGORIES)

# Training set iterator.
#train_iter = training_set[0].shuffle(1000).batch(BATCH_SIZE).make_one_shot_iterator()

# Eval set iterator.
#eval_iter = training_set[1].batch(1).make_one_shot_iterator()

def train_set():
    return training_set[0].repeat().shuffle(1000).batch(BATCH_SIZE).make_one_shot_iterator().get_next()

def eval_set():
    # return training_set[1].shuffle(1000).batch(BATCH_SIZE).make_one_shot_iterator().get_next()
    return training_set[1].shuffle(1000).batch(1).make_one_shot_iterator().get_next()




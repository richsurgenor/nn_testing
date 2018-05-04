# Could include damage % columns in csv... TODO.
# Currently only uses the building condition column.

import sys
sys.path.append('../')
import csv
import numpy
import tensorflow as tf
import pandas as pd
import home_global_variables
from PIL import Image
import time
import pprint
import random

# Percentage of eval vs training.
PERCENT_EVAL = 0.1

# Batch size constant.
BATCH_SIZE = 4

# Set image width and height.
IMG_X = 224
IMG_Y = 224

# Number of damage levels.
NUM_CLASSES = 5

# Stores the images with their path as key and their building condition as val.
train_image_label_dict = {}
eval_image_label_dict = {}


def create_image_set_map():
    data_file = pd.read_csv(home_global_variables.CSV_DIRECTORY 
                            + home_global_variables.CSV_NAME, sep=',')
    
    # Save only the image name and the building condition.
    relevant_cols = data_file[["fulcrum_id", "photos",
                               "overall_building_condition"]]
    
    # Used to help split data into training and evaluation.
    loop_count = 0
    
    # Number of rows in the pandas data frame.
    num_rows = relevant_cols.shape[0]
    print("num_rows = " + str(num_rows) + "\n\n")
    
    # Parse rows.
    for row in relevant_cols.itertuples():
        photos = row[2]
        for pic_name in photos.split(','):
            # Split into eval and training sets.
            
            # Depending on the fraction wanted for training, places certain 
            # multiples of images into eval and others into training.
            if ((loop_count % (PERCENT_EVAL * num_rows)) == 0):
                # Key on image path, value is overall building condition.
                eval_image_label_dict[home_global_variables.IMAGES_DIRECTORY + "/" + 
                                 row[1] + "/" + pic_name + ".jpg"] = row[3]
            else:
                train_image_label_dict[home_global_variables.IMAGES_DIRECTORY + "/" + 
                                 row[1] + "/" + pic_name + ".jpg"] = row[3]
             
            loop_count += 1
        
        
def _map_fn(image_path, label):
    # Decode and resize image.
    image_data = tf.read_file(image_path)
    decoded_image = tf.image.decode_jpeg(image_data, channels=3)
    image_resize = tf.image.resize_images(decoded_image, [IMG_X,IMG_Y])
    
    # Turn label into one hot vector.
    #onehot_label = tf.cast(tf.one_hot(label,NUM_CLASSES,on_value=label), tf.float32)
    onehot_label = tf.cast(tf.one_hot(label,NUM_CLASSES), tf.float32)
    
    return image_resize, onehot_label


def generate_image_sets(train_dict, eval_dict):
    
    training_dataset = tf.data.Dataset.from_tensor_slices(
        (list(train_dict.keys()),
        list(train_dict.values()))
    )
    
    evaluation_dataset = tf.data.Dataset.from_tensor_slices(
        (list(eval_dict.keys()),
        list(eval_dict.values()))
    )
    
    training_dataset = training_dataset.map(_map_fn)
    evaluation_dataset = evaluation_dataset.map(_map_fn)
    
    return training_dataset, evaluation_dataset


def label_normalize():
    # Initialize the image map if it hasn't been done yet.
    if (len(train_image_label_dict) == 0):
        create_image_set_map()

    # Create a copy of the eval and train image dictionaries for modification.
    train_dict_copy = train_image_label_dict.copy()
    eval_dict_copy = eval_image_label_dict.copy()

    #print(len(eval_dict_copy))

    count = [0] * 5

    for key in eval_dict_copy:
        count[eval_dict_copy[key]] += 1
    
    print("Evaluation distribution (unnormalized): " + str(count))
    
    # Find min.
    min_val = min(count)
    
    # Go through each entry in dict and remove categories that are proportionally
    # more represented than other categories.
    
    # Must store dict in array to shuffle.
    keys = list(eval_dict_copy.keys())
    random.shuffle(keys)

    #for key in keys:
    #    if (count[eval_dict_copy[key]] > min_val):
    #        count[eval_dict_copy[key]] -= 1
    #        eval_dict_copy.pop(key)
            
    
    print("Evaluation distribution (normalized): " + str(count))
    
    count = [0] * 5
    
    for key in train_dict_copy:
        count[train_dict_copy[key]] += 1
    
    print("Training distribution (unnormalized): " + str(count))

    # Find min.
    min_val = min(count)

    # Go through each entry in dict and remove categories that are proportionally
    # more represented than other categories.
    
    # Iterate through keys in random order.
    keys = list(train_dict_copy.keys())
    random.shuffle(keys)
    
    for key in keys:
        if (count[train_dict_copy[key]] > min_val):
            count[train_dict_copy[key]] -= 1
            train_dict_copy.pop(key)

    print("Training distribution (normalized): " + str(count))


    train_iterator, eval_iterator = generate_image_sets(train_dict_copy, 
                                                        eval_dict_copy)
    
    return train_iterator, eval_iterator


def train_set():
    training_iterator = label_normalize()[0]
    
    iter = training_iterator.repeat().shuffle(100).batch(BATCH_SIZE).make_one_shot_iterator()
    
    #return train_iterator.shuffle(1000).batch(BATCH_SIZE).make_one_shot_iterator().get_next()
    return iter.get_next()

def eval_set():
    
    evaluation_iterator = label_normalize()[1]
    
    iter = evaluation_iterator.batch(1).make_one_shot_iterator()
    
    return iter.get_next()


# All for debugging - will not be run if this module is imported...
if __name__ == "__main__":
    
    create_image_set_map()
    
    print("Eval size: " + str(len(eval_image_label_dict)) + "\n")
    print("Train size: " + str(len(train_image_label_dict)) + "\n")
    
    sess = tf.InteractiveSession()
    with sess.as_default():
        iter1, iter2 = generate_image_sets(train_image_label_dict,eval_image_label_dict)
        
        iter1 = iter1.make_one_shot_iterator()
        
        next = iter1.get_next()
        
        img = Image.fromarray(numpy.asarray(tf.cast(next[0], tf.uint8).eval()), "RGB")
        img.save("resulting_image.jpg")
        
        print(next[1].eval())
    
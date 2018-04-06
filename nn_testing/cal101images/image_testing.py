import tensorflow as tf
from PIL import Image
import numpy

IMAGE_PATH_EXAMPLE = ("/home/jwd0023/Desktop/white.jpg")

IMAGE_PATH_EXAMPLE_2 = ("/home/jwd0023/Desktop/TensorFlow Stuff/Caltech101_Testing"
"/code/caltech101/research_repo/nn_testing/cal101images/101_ObjectCategories/"
"airplanes/image_0001.jpg")

IMAGE_PATH_EXAMPLE_3 = ("/home/jwd0023/Desktop/TensorFlow Stuff/Caltech101_Testing"
"/code/caltech101/research_repo/nn_testing/cal101images/101_ObjectCategories/"
"airplanes/image_0002.jpg")

IMAGE_PATH_EXAMPLE_4 = ("/home/jwd0023/Desktop/TensorFlow Stuff/Caltech101_Testing"
"/code/caltech101/research_repo/nn_testing/cal101images/101_ObjectCategories/"
"airplanes/image_0003.jpg")



IMAGE_LABEL_EXAMPLE = 1

# Used for mapping file paths to actual images. (debug copy)
def _map_fn_debug(image_path, label):
    
    NUM_CATEGORIES = 101
    
    sess = tf.InteractiveSession()
    with sess.as_default():
        
    
        # One hot encoding.
        one_hot = tf.one_hot(tf.cast(label, dtype=tf.int32), NUM_CATEGORIES)
        
        #sess = tf.InteractiveSession()
        #with sess.as_default():
        #    print(one_hot.eval())
        
        image_file = tf.read_file(image_path)
        image_decoded = tf.image.decode_jpeg(image_file, channels=1)
    
        image_reformat = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
        image_reformat = tf.image.convert_image_dtype(image_decoded, dtype=tf.int8)
        
        image_resized = tf.image.resize_image_with_crop_or_pad(image_reformat, 200, 300)
        image_resized2 = tf.reshape(image_resized, [200, 300])
        
        #img = Image.fromarray(numpy.asarray(image_decoded.eval()), "L")
        #img.save("/home/jwd0023/Desktop/test_decoded.jpg")
        
        #img = Image.fromarray(numpy.asarray(image_reformat.eval()), "L")
        #img.save("/home/jwd0023/Desktop/test_reformatted.jpg")
        
        img = Image.fromarray(numpy.asarray(image_resized2.eval()), "L")
        img.save("/home/jwd0023/Desktop/test_reformat2.jpg")
    
        print(image_decoded.eval())
        print(one_hot.eval())

    return image_resized2, one_hot

_map_fn_debug(IMAGE_PATH_EXAMPLE, IMAGE_LABEL_EXAMPLE)
_map_fn_debug(IMAGE_PATH_EXAMPLE_2, IMAGE_LABEL_EXAMPLE)
_map_fn_debug(IMAGE_PATH_EXAMPLE_3, IMAGE_LABEL_EXAMPLE)
_map_fn_debug(IMAGE_PATH_EXAMPLE_4, IMAGE_LABEL_EXAMPLE)
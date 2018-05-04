import tensorflow as tf
from PIL import Image
import numpy

TEST_IMAGE_PATH = "./my_images/055baf51-148d-41ef-8085-3bcbd4878562.jpg"

def load_image():
    jpeg_file = tf.read_file(TEST_IMAGE_PATH)
    image_tensor = tf.image.decode_jpeg(jpeg_file, channels=3, ratio=8)
    #image_tensor = tf.expand_dims(image_tensor, 0)
    
    image_tensor = tf.image.resize_images(image_tensor, [256,256])
    
    image_tensor = tf.cast(image_tensor, tf.uint8)
    
    return image_tensor

def save_image(image_in):
    
    sess = tf.InteractiveSession()
    with sess.as_default():
        #print(image_in.eval())
        img = Image.fromarray(numpy.asarray(image_in.eval()), "RGB")
        img.save("resulting_image.jpg")

if __name__ == "__main__":
    save_image(load_image())
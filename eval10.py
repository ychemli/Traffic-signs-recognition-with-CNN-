# Import `matplotlib`
import matplotlib.pyplot as plt
import random
import dataset
import skimage
from skimage import data
from skimage import transform
import tensorflow as tf
import numpy as np
import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse

test_path = 'C:/Users/yanis/Desktop/Python_Projects/traffic_signs_data/Testing/'
image_size=28
num_channels=3
validation_size=0
classes = os.listdir(test_path)
images = []

images28, labels, _, _ = dataset.load_train(test_path, image_size, classes)

# Pick 10 random images
sample_indexes = random.sample(range(len(images28)), 20)
sample_images = [images28[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

print(sample_images[9])

images = np.array(sample_images)
 
print(images[9])


#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
x_batch = images.reshape(len(sample_images), image_size,image_size,num_channels)

## Let us restore the saved model 
sess = tf.Session()
print("Step-1: Recreate the network graph. At this step only graph is created.\n")
saver = tf.train.import_meta_graph('trafic-model.meta')
print("Step-2: Now let's load the weights saved using the restore method.\n")
saver.restore(sess, tf.train.latest_checkpoint('C:/Users/yanis/Desktop/Python_Projects/traffic/'))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

## Let's feed the images to the input placeholders
x = graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0") 
y_test_images = np.zeros((1, 62)) 


### Creating the feed_dict that is required to be fed to calculate y_pred 
feed_dict_testing = {x: x_batch, y_true: y_test_images}
result=sess.run(y_pred, feed_dict=feed_dict_testing)
# results
print(sample_labels)
print(result)

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = np.argmax(sample_labels[i])
    prediction = np.argmax(result[i])
    plt.subplot(5, 4,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
             fontsize=12, color=color)
    plt.imshow(sample_images[i],  cmap="gray")

plt.show()






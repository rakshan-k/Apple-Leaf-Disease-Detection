import os
import cv2
import PIL
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from tensorflow.keras.models import Sequential, load_model, model_from_json

# NOTE reads the image and stores in a object
img1 =image.load_img('PlantDiseasesDataset/Apple/train/Apple Black rot/AppleBlackRot(1000).JPG')

# NOTE show the image
plt.imshow(img1)

# NOTE reads the mage and stores in a object
# .shape returns the dimensions of the image (height, width, channels)
cv2.imread('PlantDiseasesDataset/Apple/train/Apple Black rot/AppleBlackRot(7).JPG').shape

train_dir ='PlantDiseasesDataset/Apple/train'
test_dir = 'PlantDiseasesDataset/Apple/test'
IMG_SIZE = (256, 256)
BATCH_SIZE = 16

# NOTE rescale=1./255: This parameter specifies the scaling factor for pixel values in the image data. 
#By dividing each pixel value by 255, the pixel values will be normalized to the range of 0 to 1. 
#This rescaling is commonly done to bring the pixel values to a suitable range for training neural networks.

# NOTE validation_split=0.2: This parameter specifies the fraction of the training data that will be used for validation. 
#In this case, 20% of the training data will be reserved for validation during the training process.
#The validation data is used to evaluate the model's performance on unseen data and to monitor the model's progress during training.
train_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
    )
val_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
    )  

# NOTE batch_size=16: This parameter specifies the number of samples per batch that 
#will be generated from the directory. 
# In this case, each batch will contain 16 images.

train_set = train_gen.flow_from_directory(
    train_dir,
    subset = 'training',
    class_mode = 'categorical',
    target_size = IMG_SIZE,
    batch_size = 16
)

val_set = val_gen.flow_from_directory(
   train_dir,
   subset = 'validation',
   class_mode = 'categorical',
   target_size = IMG_SIZE,
   batch_size = 16
   )

train_set.classes # NOTE array([0, 0, 0, ..., 3, 3, 3])

train_set.class_indices # NOTE {'Apple Black rot': 0,
 #                              'Apple Healthy': 1,
 #                              'Apple Scab': 2,
 #                              'Cedar apple rust': 3}

## building the model


model = tf.keras.models.Sequential([
        
        layers.InputLayer(input_shape=(256, 256, 3)),
        # NOTE accepts input image of size 256x256 and 3 color (RGB)
    
        layers.Conv2D( 32, 3,padding='valid', activation='relu'),
        # NOTE convolutional layer with 32 fileter
        # kernel size of 3x3, valid padding
        # ReLU activation function
        
        layers.MaxPooling2D(pool_size=(2,2)),
        # NOTE max pooling with size of 2x2
        #  #########
        
        layers.Conv2D( 64, 3,padding='valid', activation='relu'),
        
        layers.MaxPooling2D(pool_size=(2,2)),
        #  #########
        
        layers.Conv2D( 64, 3,padding='valid', activation='relu'),
        
        layers.MaxPooling2D(pool_size=(2,2)),
        #  #########
        
        layers.Conv2D( 64, 3,padding='valid', activation='relu'),
        
        layers.MaxPooling2D(pool_size=(2,2)),
        #  #########
        
        layers.Conv2D( 128, 3,padding='valid', activation='relu'),
        
        layers.MaxPooling2D(pool_size=(2,2)),
        # ##########
        
        layers.Conv2D( 64, 3,padding='valid', activation='relu'),
        
        layers.MaxPooling2D(pool_size=(2,2)),
        # ##########
        
        layers.Flatten(),
        # NOTE flattens the output of previous into a 1D array 
        # prepares the data for input to the dense layer
        
        layers.Dense(64, activation='relu'),
        # NOTE This dense layer has 64 units and ReLU activation. 
        #It is a fully connected layer that applies a 
        #linear transformation to the input data.
        
        layers.Dense(4, activation='softmax')
        ])
        # NOTE This dense layer is the output layer of
        # the model with 4 units and softmax activation.
        # It produces a probability distribution over the
        # 4 possible classes, indicating the predicted 
        #probabilities for each class.

print(model.summary())

# NOTE loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False):
# This line specifies the loss function to be used during training.
# In this case, CategoricalCrossentropy is used, which is suitable 
# for multi-class classification problems. The from_logits=False
# argument indicates that the model's output is already converted 
# into probabilities using softmax activation.

# optimizer=keras.optimizers.Adam(learning_rate=1e-3):
# This line defines the optimizer to be used during training.
# Here, the Adam optimizer is used, which is a popular choice for 
# gradient-based optimization algorithms. The learning_rate parameter
# is set to 1e-3 (0.001), which determines the step size at each iteration.

# metrics=['accuracy']: This line specifies the metric(s) to
# be evaluated during training and testing. In this case, accuracy 
# is used as the metric, which measures the proportion of correctly classified samples.
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=['accuracy']
    )

# fittiing the images into the model

final_model = model.fit(
     train_set,
     epochs=15, # NOTE no. of iteration
     validation_data=val_set, 
     steps_per_epoch = len(train_set),
     validation_steps = len(val_set)
     )

test_gen = ImageDataGenerator(rescale=1./255) # NOTE normalize the pixel values to the range of [0, 1]

test_set = test_gen.flow_from_directory(
    test_dir,
    class_mode = 'categorical',
    target_size = IMG_SIZE,
    batch_size = 16
)

final_model.params
# NOTE returns a dictionary that contains the
# parameters and settings of the trained model.
#number of samples seen, number of epochs, batch size,
# steps per epoch, validation steps, 

final_model.history.keys()
# NOTE returns a dictionary that contains 
# the history of the training process

acc = final_model.history['accuracy']
val_acc = final_model.history['val_accuracy']

loss = final_model.history['loss']
val_loss = final_model.history['val_loss']

# plot the accuracy and loss
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(15), acc, label='Training Accuracy')
plt.plot(range(15), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.show()
# plt.savefig('AccVal_acc')

plt.subplot(1, 2, 2)
plt.plot(range(15), loss, label='Training Loss')
plt.plot(range(15), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
# plt.savefig('LossVal_loss')

model.save('./apple_leaf_cnn_model.h5')

## making predictions with test images

def predict(model, images):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i])
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = test_set.classes[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    return predicted_class, confidence

scores = model.evaluate(test_set, batch_size=64, verbose=2)

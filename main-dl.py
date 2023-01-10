print("Importing libraries...")
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib 
import time
#print(device_lib.list_local_devices())

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#print(device_lib.list_local_devices())
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#print(tf.config.list_physical_devices('GPU'))

# Load the training data and labels
print('Loading training data...')
X_train = []
y_train = []

Train_Data_Fire = r"D:\Code\DataSet\ForestFire\Train_Data\Fire"
Train_Data_NonFire = r"D:\Code\DataSet\ForestFire\Train_Data\Non_Fire"

#Train_Data_Fire = r'D:\Code\DataSet\ForestFire\Test_Data\Fire'
#Train_Data_NonFire = r"D:\Code\DataSet\ForestFire\Test_Data\Non_Fire"

# Load the fire images and labels
print('Loading fire images...')
for filename in os.listdir(Train_Data_Fire):
    image = cv2.imread(os.path.join(Train_Data_Fire, filename))
    X_train.append(image)
    y_train.append(1)

# Load the non-fire images and labels
print('Loading non-fire images...')
for filename in os.listdir(Train_Data_NonFire):
    image = cv2.imread(os.path.join(Train_Data_NonFire, filename))
    X_train.append(image)
    y_train.append(0)

# Convert the images to grayscale
print('Converting images to grayscale...')
X_train_gray,c = [],0
for counter in range(len(X_train)):
    image = X_train[counter]
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        X_train_gray.append(gray)
    except:
        del y_train[counter]
        print('Error occurred while converting image to grayscale')
        continue

#Resize the images to a uniform size

print('Resizing images to uniform size...')
X_train_resized = []
for image in X_train_gray:
    resized_image = cv2.resize(image, (100, 100))
    X_train_resized.append(resized_image)

#Convert the images and labels to numpy arrays

X_train_resized = np.array(X_train_resized)
y_train = np.array(y_train)

#Normalize the pixel values

print('Normalizing pixel values...')
X_train_normalized = X_train_resized / 255.0

#Build the model

print('Building the model...')
model = keras.Sequential([
keras.layers.Flatten(input_shape=(100, 100)),
keras.layers.Dense(512, activation='relu'),
keras.layers.Dense(2, activation='softmax')
])

#Compile the model

print('Compiling the model...')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Train the model

print('Training the model...')
model.fit(X_train_normalized, y_train, epochs=200)

# Save the model
print('Saving the model...')
model.save(r"D:\Code\Python\SDG\Forest_Fire_Detector\Model\my_model_l-512_E-200.h5")
print('Model saved successfully')

time.sleep(1000)

'''
#Load the new image to be predicted
print('Loading the new image to be predicted...')
new_image = cv2.imread(r"D:\Code\DataSet\ForestFire\Train_Data\Fire\F(2).jpg")

#Convert the new image to grayscale and resize it

print('Converting and resizing the new image...')
new_image_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
new_image_resized = cv2.resize(new_image_gray, (100, 100))

#Normalize the pixel values of the new image

print('Normalizing the pixel values of the new image...')
new_image_normalized = new_image_resized / 255.0

#Use the model to predict whether the new image is a forest fire

print('Predicting whether the new image is a forest fire...')
prediction = model.predict(np.array([new_image_normalized]))
prediction_class = np.argmax(prediction)

print(prediction_class)

if prediction_class >= 0.5:
    print('This is a forest fire')
else:
    print('This is not a forest fire')

    '''
#!/usr/bin/env python
# coding: utf-8

# In[1]:













pip install tensorflow matplotlib numpy


# In[2]:


# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

# Load pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

# Function to preprocess images
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to extract image features using ResNet50
def extract_image_features(image_path):
    img_array = preprocess_image(image_path)
    features = model.predict(img_array)
    return features

# Load and preprocess captions
captions = ["startseq dog playing in the grass endseq", "startseq cat on the roof endseq", ...]  # Add your captions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1

# Create sequences of tokens from captions
sequences = tokenizer.texts_to_sequences(captions)

# Create input-output pairs for training
X, y = [], []
for seq in sequences:
    for i in range(1, len(seq)):
        in_seq, out_seq = seq[:i], seq[i]
        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        X.append(in_seq)
        y.append(out_seq)

X, y = np.array(X), np.array(y)

# Define the image captioning model
image_input = Input(shape=(2048,))
image_dense = Dense(256, activation='relu')(image_input)

caption_input = Input(shape=(max_length,))
caption_embedding = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(caption_input)
caption_lstm = LSTM(256)(caption_embedding)

decoder_input = tf.concat([image_dense, caption_lstm], axis=-1)
decoder_output = Dense(vocab_size, activation='softmax')(decoder_input)

model = Model(inputs=[image_input, caption_input], outputs=decoder_output)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit([image_features, X], y, epochs=10, batch_size=64)

# Save the model
model.save("image_captioning_model.h5")

# Function to generate captions for new images
def generate_caption(model, image_path):
    image_features = extract_image_features(image_path)
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Function to map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Example usage
new_image_path = "path/to/your/image.jpg"
caption = generate_caption(model, new_image_path)
print("Generated Caption:", caption)


# In[1]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

# Function to preprocess images
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to extract image features using ResNet50
def extract_image_features(image_path):
    img_array = preprocess_image(image_path)
    features = model.predict(img_array)
    return features

# Load and preprocess captions
captions = [
    "startseq dog playing in the grass endseq",
    "startseq cat on the roof endseq"
    # Add your captions here
]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in captions)

# Create sequences of tokens from captions
sequences = tokenizer.texts_to_sequences(captions)

# Create input-output pairs for training
X, y = [], []
for seq in sequences:
    for i in range(1, len(seq)):
        in_seq, out_seq = seq[:i], seq[i]
        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        X.append(in_seq)
        y.append(out_seq)

X, y = np.array(X), np.array(y)

# Assuming you have a list of training image paths
train_image_paths = ["C:\\Users\\91701\\Downloads\\IMG-20230720-WA0000.jpg", "C:\\Users\\91701\\Downloads\\IMG-20230720-WA0000.jpg"]
# Extract features for all training images
image_features_list = [extract_image_features(path) for path in train_image_paths]

# Stack the features into a numpy array
image_features = np.vstack(image_features_list)

# Define the image captioning model
image_input = Input(shape=(2048,))
image_dense = Dense(256, activation='relu')(image_input)

caption_input = Input(shape=(max_length,))
caption_embedding = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(caption_input)
caption_lstm = LSTM(256)(caption_embedding)

decoder_input = tf.concat([image_dense, caption_lstm], axis=-1)
decoder_output = Dense(vocab_size, activation='softmax')(decoder_input)

model = Model(inputs=[image_input, caption_input], outputs=decoder_output)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit([image_features, X], y, epochs=10, batch_size=64)

# Save the model
model.save("image_captioning_model.h5")

# Function to generate captions for new images
def generate_caption(model, image_path):
    image_features = extract_image_features(image_path)
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Function to map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Example usage
new_image_path = "C:\\Users\\91701\\Downloads\\IMG-20230720-WA0000.jpg"
caption = generate_caption(model, new_image_path)
print("Generated Caption:", caption)


# In[2]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

# Function to preprocess images
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to extract image features using ResNet50
def extract_image_features(image_path):
    img_array = preprocess_image(image_path)
    features = model.predict(img_array)
    return features

# Assuming you have a list of training image paths
train_image_paths = ["C:\\Users\\91701\\Downloads\\IMG-20230720-WA0000.jpg", "C:\\Users\\91701\\Downloads\\IMG-20230720-WA0000.jpg"]

# Extract features for all training images
image_features_list = [extract_image_features(path) for path in train_image_paths]

# Stack the features into a numpy array
image_features = np.vstack(image_features_list)

# Assuming you have a list of training captions
captions = [
    "startseq dog playing in the grass endseq",
    "startseq cat on the roof endseq"
    # Add your captions here
]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in captions)

# Create sequences of tokens from captions
sequences = tokenizer.texts_to_sequences(captions)

# Create input-output pairs for training
X, y = [], []
for seq in sequences:
    for i in range(1, len(seq)):
        in_seq, out_seq = seq[:i], seq[i]
        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        X.append(in_seq)
        y.append(out_seq)

X, y = np.array(X), np.array(y)

# Check if the number of samples match
assert len(image_features) == len(X), "Number of samples in image_features and X do not match"

# Define the image captioning model
image_input = Input(shape=(2048,))
image_dense = Dense(256, activation='relu')(image_input)

caption_input = Input(shape=(max_length,))
caption_embedding = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(caption_input)
caption_lstm = LSTM(256)(caption_embedding)

decoder_input = tf.concat([image_dense, caption_lstm], axis=-1)
decoder_output = Dense(vocab_size, activation='softmax')(decoder_input)

model = Model(inputs=[image_input, caption_input], outputs=decoder_output)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit([image_features, X], y, epochs=10, batch_size=64)

# Save the model
model.save("image_captioning_model.h5")

# Function to generate captions for new images
def generate_caption(model, image_path):
    image_features = extract_image_features(image_path)
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Function to map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Example usage
new_image_path = "C:\\Users\\917+
01\\Downloads\\IMG-20230720-WA0000.jpg"
caption = generate_caption(model, new_image_path)
print("Generated Caption:", caption)


# In[3]:


import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Build a simple neural network model
model = Sequential()
model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test Accuracy:", test_acc)

# Save the model
model.save("mnist_classifier_model.h5")


# In[4]:


import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load and preprocess the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Build a convolutional neural network model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test Accuracy:", test_acc)

# Save the model
model.save("emergency_image_classifier_model.h5")


# In[ ]:





# In[9]:


import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model("emergency_image_classifier_model.h5")

# Path to your test image
test_image_path = "C:\\Users\\91701\\Downloads\\archive (3)\\Emergency_Vehicles\\test"

# Load and preprocess the test image
img = image.load_img(test_image_path, target_size=(32, 32))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalize pixel values

# Make predictions
predictions = model.predict(img_array)

# Get the class label (emergency or non-emergency)
class_label = "Emergency Vehicle" if predictions[0][0] > 0.5 else "Non-Emergency Vehicle"

print(f"Prediction: {class_label}")


# In[14]:


pip install flickrapi


# In[18]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define paths to the training dataset
train_data_dir = 'path/to/your/dataset'

# Image data generators for training
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)

# Load and augment training data
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(64, 64), batch_size=32, class_mode='binary', subset='training')

# Image data generator for validation (if you have a validation set)
validation_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(64, 64), batch_size=32, class_mode='binary', subset='validation')

# Build a simple CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Save the model
model.save("emergency_vehicle_detection_model.h5")

# Load the trained model
model = tf.keras.models.load_model("emergency_vehicle_detection_model.h5")

# Path to your test image
test_image_path = "path/to/your/test_image.jpg"

# Load and preprocess the test image
img = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(64, 64))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalize pixel values

# Make predictions
prediction = model.predict(img_array)

# Get the class label (emergency or non-emergency)
class_label = "Emergency Vehicle" if prediction[0][0] > 0.5 else "Non-Emergency Vehicle"

print(f"Prediction: {class_label}")


# In[19]:


train_df = pd.read_csv("C:\\Users\\91701\\Downloads\\archive (4)\\train_SOaYf6m\\train.csv")
test_df = pd.read_csv("C:\\Users\\91701\\Downloads\\archive (4)\\test_vc2kHdQ.csv")
submit = pd.read_csv("C:\\Users\\91701\\Downloads\\archive (4)\\sample_submission_yxjOnvz.csv")
train_df.shape, test_df.shape
train_df.groupby('emergency_or_not').count()
sns.countplot(x='emergency_or_not' , data=train_df)
data_folder = Path("C:\\Users\\91701\\Downloads\\archive (4)")
data_path = "C:\\Users\\91701\\Downloads\\archive (4)\\train_SOaYf6m\\images"

path = os.path.join(data_path , "*jpg")
data_path

files = glob.glob(path)
data=[]
for file in files:
    image = cv2.imread(file)
    data.append(image)
    
train_images = data[:1646]
test_images= data[1646:]

print(train_images[0].shape), print(train_images[100].shape)

def get_images_class(cat):
    list_of_images = []
    fetch = train_df.loc[train_df['emergency_or_not']== cat][:3].reset_index()
    for i in range(0,len(fetch['image_names'])):
        list_of_images.append(fetch['image_names'][i])
    return list_of_images 
get_images_class(0)
get_images_class(1)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
fig = plt.figure(figsize=(20,15))
for i, image_name in enumerate(get_images_class(0)):
    plt.subplot(1,3 ,i+1)
    img=mpimg.imread("C:\\Users\\91701\\Downloads\\archive (4)\\train_SOaYf6m\\images\\"+image_name)
    imgplot = plt.imshow(img)
    plt.xlabel(str("Non-Emergency Vehicle") + " (Index:" +str(i+1)+")" )
plt.show()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
fig = plt.figure(figsize=(20,15))
for i, image_name in enumerate(get_images_class(1)):
    plt.subplot(1,3 ,i+1)
    img=mpimg.imread("C:\\Users\\91701\\Downloads\\archive (4)\\train_SOaYf6m\\images\\"+image_name)
    imgplot = plt.imshow(img)
    plt.xlabel(str("Emergency Vehicle") + " (Index:" +str(i)+")" )
plt.show()


# In[ ]:





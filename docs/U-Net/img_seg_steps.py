import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the data
images = np.load('images.npy')
masks = np.load('masks.npy')

# Normalize the images
images = images / 255.0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

# Split *that* data into validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define data generators to feed the data to the model during training
train_datagen = ImageDataGenerator(rotation_range=90,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True)

# iterator
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)

val_datagen = ImageDataGenerator()
val_generator = val_datagen.flow(X_val, y_val, batch_size=32)

test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow(X_test, y_test, batch_size=32)

# Define the input shape
input_shape = (256, 256, 3)

# Define the encoder
inputs = Input(shape=input_shape)
conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
drop4 = Dropout(0.5)(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

# Define the decoder
up5 = UpSampling2D(size=(2, 2))(pool4)
up5 = Conv2D(512, 2, activation='relu', padding='same')(up5)
merge5 = Concatenate()([drop4, up5])
conv5 = Conv2D(512, 3, activation='relu', padding='same')(merge5)
conv5 = Conv2D(512, 3, activation='relu', padding='same')(conv5)

up6 = UpSampling2D(size=(2, 2))(conv5)
up6 = Conv2D(256, 2, activation='relu', padding='same')(up6)
merge6 = Concatenate()([conv3, up6])
conv6 = Conv2D(256, 3, activation='relu', padding='same')(merge6)
conv6 = Conv2D(256, 3, activation='relu', padding='same')(conv6)

up7 = UpSampling2D(size=(2, 2))(conv6)
up7 = Conv2D(128, 2, activation='relu', padding='same')(up7)
merge7 = Concatenate()([conv2, up7])
conv7 = Conv2D(128, 3, activation='relu', padding='same')(merge7)
conv7 = Conv2D(128, 3, activation='relu', padding='same')(conv7)

up8 = UpSampling2D(size=(2, 2))(conv7)
up8 = Conv2D(64, 2, activation='relu', padding='same')(up8)
merge8 = Concatenate()([conv1, up8])
conv8 = Conv2D(64, 3, activation='relu', padding='same')(merge8)
conv8 = Conv2D(64, 3, activation='relu', padding='same')(conv8)

num_classes = np.unique(y_train).shape[0]  # I'm guessing.
input_layer = Input(shape=[256, 256, 3])  # Here too.

# Output layer
output_layer = Conv2D(num_classes, kernel_size=(1, 1), activation='softmax')(conv8)

# Define the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# TODO: what are the inputs here?
train_images = []
train_masks = []
val_images = []
val_masks = []
test_images = []
test_masks = []

# Train the model
model.fit(train_images, train_masks, batch_size=16, epochs=50, validation_data=(val_images, val_masks))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_masks)


def load_new_images():
    """
    I think you make this yourself.
    """
    return "idk"


def visualize_segmentation(a, b):
    """
    Hello matplotlib.
    """
    pass


# Make predictions on new images
new_images = load_new_images()
predictions = model.predict(new_images)

# Visualize the segmentation masks
for i in range(len(predictions)):
    visualize_segmentation(new_images[i], predictions[i])

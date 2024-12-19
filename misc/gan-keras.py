"""
Trains a Generative Adversarial Network (GAN) to generate synthetic images of MNIST handwritten digits.
"""
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, LeakyReLU, BatchNormalization, Input
from keras.models import Sequential, Model
from keras.optimizers import Adam

# Define the generator network
latent_dim = 100  # Dimension of the noise vector
generator = Sequential([
    Dense(128, input_dim=latent_dim),
    LeakyReLU(alpha=0.01),
    BatchNormalization(),
    Dense(784, activation='tanh')
])

# generator.compile(loss='binary_crossentropy', optimizer='adam')
generator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

# Define the discriminator network
discriminator = Sequential()
discriminator.add(Dense(128, input_dim=784))  # one-dimensional array of length 784 (28 x 28) pixels
discriminator.add(LeakyReLU(alpha=0.01))
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

# Combine the generator and discriminator networks to form the GAN
z = Input(shape=(100,))
img = generator(z)
discriminator.trainable = False
validity = discriminator(img)
gan = Model(z, validity)
gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

# Load the dataset
(X_train, _), (_, _) = mnist.load_data()

# Normalize and flatten the images
X_train = X_train / 127.5 - 1.
X_train = np.reshape(X_train, (X_train.shape[0], 784))

# Train the GAN
# epochs = 10000  # todo: Train it 10K times.
epochs = 200
batch_size = 128
for epoch in range(epochs):
    # Generate random noise
    noise = np.random.normal(0, 1, (batch_size, 100))

    # Generate fake images
    gen_imgs = generator.predict(noise)

    # Train the discriminator on real and fake images
    d_loss_real = discriminator.train_on_batch(X_train[np.random.randint(0, X_train.shape[0], batch_size)],
                                               np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train the generator to fool the discriminator
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    # Print the losses every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")

import matplotlib.pyplot as plt


# Function to generate and visualize synthetic MNIST digit images
def generate_and_visualize(generator, num_samples=16):
    """
    Generates synthetic MNIST digit images using the generator and visualizes them with matplotlib.
    :param generator: Trained generator model.
    :param num_samples: Number of samples to generate and display (default is 16).
    """
    noise = np.random.normal(0, 1, (num_samples, latent_dim))  # Random noise input
    generated_images = generator.predict(noise)

    # Rescale images to [0, 1] for visualization
    generated_images = 0.5 * generated_images + 0.5
    generated_images = generated_images.reshape(num_samples, 28, 28)  # Reshape to 28x28

    # Plot the images
    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    cnt = 0
    for i in range(4):
        for j in range(4):
            axs[i, j].imshow(generated_images[cnt], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    plt.show()


# Call the visualization function after training the generator
# Assuming the generator has been trained
generate_and_visualize(generator)

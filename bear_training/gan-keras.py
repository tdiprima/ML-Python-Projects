"""
GAN.md
Deep Fake Using GAN
Epoch: 0, Discriminator Loss: 0.6365200877189636, Generator Loss: 0.7656040191650391
"""
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, LeakyReLU, BatchNormalization, Input
from keras.models import Sequential, Model
from keras.optimizers import Adam

# Define the generator network
generator = Sequential()
generator.add(Dense(128, input_dim=100))
generator.add(LeakyReLU(alpha=0.01))
generator.add(BatchNormalization())
generator.add(Dense(784, activation='tanh'))

# todo: lr deprecated, use learning_rate
# generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
generator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

# Define the discriminator network
discriminator = Sequential()
discriminator.add(Dense(128, input_dim=784))
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
# epochs = 10000 # todo: This is the one.
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

    # TODO: output & display

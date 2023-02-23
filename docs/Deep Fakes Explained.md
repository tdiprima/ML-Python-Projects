## Deep Fakes Explained 🧙‍♂️ 🪄 🎩

Have you ever seen a magic trick where a magician makes something **disappear** or someone **appear** out of nowhere? Well, deep fakes are kind of like a magic trick, but with **videos**. 📺

Deep fakes are videos that have been made to look like **they show something that actually didn't happen**.

For example, someone could use special computer programs to make a video of someone saying or doing something that they never actually said or did.

These videos can be really convincing and look very real, which can be a problem because some people might believe them and think that they're true. That's why it's important to always be careful and try to make sure that the things you see and hear online are actually true.

## Creating Deep Fakes with GANs.

Deep Fakes are typically created using deep learning algorithms, specifically a type of **neural network** known as a **Generative Adversarial Network (GAN)**. GANs consist of two parts: 

1. Generator network
2. Discriminator network

### Generator network 🙂 => 🐶

The generator network takes in an **input image** and **generates a new image** based on the characteristics of a dataset of **target images**.

*(I think the targets is kinda like when put the little balls on the stunt people's outfits, and voila - you can doctor up their outfits with CGI or make them look like an animal.  Except instead of using film, you might have still-shots instead. Maybe.)*

### Discriminator network 🎞️ 🔍

The discriminator network tries to distinguish between the **generated image** and the **real image** from the target dataset. 

💡The generator network is **trained to fool** the discriminator network into thinking that the generated image is real, while the discriminator network is trained to correctly identify the real images.

By training the GAN on a large dataset of target images, the generator network can **learn to create realistic images** that resemble the target images.

This can be used to create deep fakes by feeding in an image of a person and generating a new image that appears to be that person, but with different actions or expressions.

There are also other **deep learning algorithms** used in the creation of Deep Fakes, such as:

* Autoencoders
* Variational autoencoders

which are used to generate latent representations of faces or other objects that can be manipulated to create new images.

## Latent? 🤔

When generating latent representations of faces, we are referring to creating a lower-dimensional vector that captures the **essential characteristics** or features of a face.

These features are typically learned by training an autoencoder or variational autoencoder on a large dataset of faces.

### Auto-encoder 🗺️ 

An autoencoder is a type of neural network that is trained to encode an input image into a **lower-dimensional representation**, and then decode it back into the original image.

During training, the autoencoder learns to compress the input image into a compressed representation that **contains the most important features** of the image.

This compressed representation is known as the "latent space" or "latent code", and it **can be used to generate new images** that share the same features as the input image.

### Variational autoencoder

A variational autoencoder (VAE) is a type of autoencoder that has a different architecture and training objective.

In addition to encoding and decoding the input image, a VAE also learns to generate new images by **sampling from a probability distribution** over the latent space.

This allows the VAE to generate new images that are **similar** to the training images, but also have some **variation**.

In the case of faces, the latent representation might capture features such as 

* the shape of the face
* the position and size of the 
    * eyes
    * nose
    * mouth
* and other facial features

This latent representation can be used to generate new faces with different attributes, such as **different expressions** or **poses**, by modifying the values in the latent space.

## Code

A **simple GAN** in Python using the **Keras** deep learning library:

```py
# Import necessary libraries
from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, BatchNormalization, Input
from keras.optimizers import Adam
import numpy as np

# Define the generator network
generator = Sequential()
generator.add(Dense(128, input_dim=100))
generator.add(LeakyReLU(alpha=0.01))
generator.add(BatchNormalization())
generator.add(Dense(784, activation='tanh'))
generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

# Define the discriminator network
discriminator = Sequential()
discriminator.add(Dense(128, input_dim=784))
discriminator.add(LeakyReLU(alpha=0.01))
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

# Combine the generator and discriminator networks to form the GAN
z = Input(shape=(100,))
img = generator(z)
discriminator.trainable = False
validity = discriminator(img)
gan = Model(z, validity)
gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

# Load the dataset
(X_train, _), (_, _) = mnist.load_data()

# Normalize and flatten the images
X_train = X_train / 127.5 - 1.
X_train = np.reshape(X_train, (X_train.shape[0], 784))

# Train the GAN
epochs = 10000
batch_size = 128
for epoch in range(epochs):
    # Generate random noise
    noise = np.random.normal(0, 1, (batch_size, 100))
    
    # Generate fake images
    gen_imgs = generator.predict(noise)
    
    # Train the discriminator on real and fake images
    d_loss_real = discriminator.train_on_batch(X_train[np.random.randint(0, X_train.shape[0], batch_size)], np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    # Train the generator to fool the discriminator
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
    
    # Print the losses every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")

```

<br>
This code defines a simple GAN that can **generate MNIST digits.**

The generator and discriminator networks are defined using the **Keras Sequential model**.

The generator takes in random noise and **generates a fake image**, while the discriminator takes in an image and **outputs a binary value** indicating whether the image is real or fake.

The GAN is trained by alternating between training the discriminator on real and fake images, and training the generator to fool the discriminator.

The losses for both the discriminator and generator are printed every 100 epochs.

### What?

Okay, let me try to explain it in a simpler way.

Imagine you have a game where you draw pictures of animals, and your friend has to guess whether the animal is real or made up.

**(Balderdash.)**

In this game, you are the generator and your friend is the discriminator. Your job as the generator is to draw pictures of animals, some of which are real and some of which are made up. The discriminator's job is to guess whether each picture you draw is a real animal or a made-up one.

When you play the game, you take turns: first you draw a picture of an animal and show it to your friend, who guesses whether it's real or fake. Then you draw another picture, and your friend guesses again.

Now imagine that you are trying to get better at this game, so you practice a lot. You draw lots of pictures, and your friend guesses whether they are real or fake. As you practice, you get better at drawing animals that look more and more like real ones.

In the same way, a GAN works by having two parts: a **generator** that creates fake images, and a **discriminator** that tries to tell the difference between real and fake images.

The GAN ***(the both of ya's)*** is trained by having the generator create fake images, which the discriminator tries to identify as fake.

Then the **generator learns** from the feedback it gets from the discriminator, and tries to create better fake images that fool the discriminator more effectively.

This process continues, with the generator and discriminator taking turns, until the generator can **create images** that are **indistinguishable from real ones.**

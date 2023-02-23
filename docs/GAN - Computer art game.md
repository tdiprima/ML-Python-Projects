## Generative adversarial networks

Have you ever played a game of "guess who" with a friend? In that game, you each have a board with different characters on it, and you take turns asking **yes-or-no questions** to try and guess the other person's character.

Now imagine that instead of a friend, you're playing against a computer. But instead of playing "guess who," you're trying to make **pictures of animals.** The computer will show you a picture of an animal, and you have to draw your own picture of the same animal. The computer will then compare your picture to the original picture and tell you how close you came.

But the computer isn't just trying to help you make better pictures. It's also trying to make its own pictures of animals that are so good, you won't be able to tell the difference between them and real pictures. The computer will keep making better and better pictures, and you'll keep trying to catch up.

This is kind of like what happens in something called a "generative adversarial network," or GAN for short. It's a special kind of computer program that has two parts: one part **makes pictures**, and the other part tries to **figure out** which pictures are real and which ones are fake.

⭐️ The part that makes the pictures tries to make them so good that the other part can't tell the difference.

It's like a game between two players, where one player is trying to make really good fake pictures, and the other player is trying to catch them out by **spotting the fakes.** The two players keep playing this game over and over again, and each time they get better and better. The end result is a computer program that's really good at **making realistic-looking pictures!**

## Deep Fake Using GAN

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

## Ok, how do I display the results?

The **output** of a Generative Adversarial Network (GAN) can vary depending on the specific task and architecture of the network. 

However, in general, the output of a GAN is a **generated image** that is created by the generator model of the network.

After training your GAN in Keras, you can display the results by **generating some new images** using the trained generator model.

One way to do this is by using the **`predict` method** of the generator model. This method takes a random noise vector as input and generates an image as output.

Here's some sample code that shows how you could generate an image using the trained generator model:

```python
# generate a random noise vector
noise = np.random.normal(0, 1, size=(1, latent_dim))

# use the generator model to generate an image
generated_image = generator.predict(noise)

# display the generated image
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.axis('off')
plt.show()
```

<br>
In this code, **`latent_dim`** is the **size of the noise vector** used as input to the generator model.

The **`predict`** method of the generator model takes this noise vector as input and generates an image.

Finally, the generated image is displayed using the **imshow** function from matplotlib.

## Generative adversarial networks

<span style="font-size: 30px;">⚙️ 🥊 🌐</span>

GAN (Generative Adversarial Network) is a type of neural network architecture that consists of two sub-networks: a generator network and a discriminator network.

The generator network creates fake data samples that try to mimic the real data, while the discriminator network tries to distinguish between the real and fake samples.

The two networks are trained together in an adversarial manner, with the generator trying to fool the discriminator and the discriminator trying to correctly identify the fake samples.


### Deep Fake Using GAN

<span style="color:#0000dd;font-size:larger;">See: gan-keras.py</span>

This code defines a simple GAN that can generate MNIST digits.

The generator and discriminator networks are defined using the Keras Sequential model.

The generator takes in random noise and generates a fake image, while the discriminator takes in an image and outputs a binary value indicating whether the image is real or fake.

The GAN is trained by alternating between training the discriminator on real and fake images, and training the generator to fool the discriminator.

The losses for both the discriminator and generator are printed every 100 epochs.

### How to display the results

The output of a Generative Adversarial Network (GAN) can vary depending on the specific task and architecture of the network. 

However, in general, the output of a GAN is a generated image that is created by the generator model of the network.

After training your GAN in Keras, you can display the results by generating some new images using the trained generator model.

One way to do this is by using the `predict` method of the generator model. This method takes a random noise vector as input and generates an image as output.

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
In this code, `latent_dim` is the size of the noise vector used as input to the generator model.

The `predict` method of the generator model takes this noise vector as input and generates an image.

Finally, the generated image is displayed using the imshow function from matplotlib.

<br>

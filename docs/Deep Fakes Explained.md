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

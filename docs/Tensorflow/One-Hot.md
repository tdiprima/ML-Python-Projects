## one\_hot

In machine learning, one-hot encoding is a process of representing categorical variables as binary vectors, 
where each vector has a single element set to 1, and all other elements set to 0.

For example, suppose we have a dataset of animal types, with categories like "cat", "dog", and "bird". 

We can encode these categories using one-hot encoding, like this:

```css
cat -> [1, 0, 0]
dog -> [0, 1, 0]
bird -> [0, 0, 1]
```

<br>

In this encoding, each category is represented by a binary vector of length 3, where the position of the 1 indicates the category.

One-hot encoding is useful for machine learning algorithms that expect numerical inputs, because it provides 
a way to represent categorical data in a way that is compatible with these algorithms. 

For example, in a neural network, we might use a one-hot encoded vector as the output of the network, 
where each element of the vector corresponds to a different **class**, and the element with the **highest value** indicates the **predicted** class.

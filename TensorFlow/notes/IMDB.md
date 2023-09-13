## Smallest Dataset

<span style="color:#0000dd;">Which is the smallest dataset out of keras's boston\_housing, cifar10, cifar100, fashion\_mnist, imdb, mnist, reuters?</span>

Among the datasets you listed, the smallest dataset is `imdb`, which stands for the Internet Movie Database. It is a dataset used for sentiment analysis, containing movie reviews and their associated sentiment labels (positive or negative).

The size of the `imdb` dataset is relatively small compared to the other datasets you mentioned. It contains a total of 50,000 movie reviews, split into 25,000 reviews for training and 25,000 reviews for testing.

Here's how you can load the `imdb` dataset using TensorFlow's Keras API:

```python
import tensorflow as tf
from tensorflow.keras.datasets import imdb

# Load the IMDB dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
```

<br>

The `num_words=10000` argument limits the dataset to the top 10,000 most frequently occurring words in the reviews. This is done to keep the dataset manageable while preserving most of the relevant information for sentiment analysis.

## IMDb Features

<span style="color:red;">question - so what are these 4 features in the imdb dataset?</span>

*I was wrong &ndash; 4 was the Iris dataset.*

In the IMDb dataset, each data point (movie review) consists of a sequence of words from the review's text. Each word is represented by an integer index, which corresponds to the word's position in a word index dictionary.

The IMDb dataset does not have traditional features like numerical or categorical columns, as in other datasets. Instead, the features are sequences of integers, where each integer represents a specific word in the movie review.

To summarize, the four features in the IMDb dataset are:

1. Integer sequences representing the words in the movie review.
2. Each integer represents a specific word from the word index dictionary.
3. The sequence length may vary from one review to another.
4. The `num_words` argument (e.g., `num_words=10000`) during loading limits the dataset to the top N most frequently occurring words, where N is the value of `num_words`. This is done to manage the dataset size and remove less common words.

The target variable (label) is a binary sentiment value: 0 for negative sentiment and 1 for positive sentiment. The goal is to predict the sentiment (positive or negative) based on the sequence of words in the review.

<br>

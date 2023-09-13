## TensorFlow in JavaScript

TensorFlow.js is a library that allows you to run machine learning models directly in the browser or in Node.js.

Here is an example of how to use TensorFlow.js to train a simple model to recognize handwritten digits:

<a href="example-1.py">example 1</a> stopped at 185 digits.

It tried it again, this time, it stopped after 245 digits.

Apparently, your array has too many digits for the browser to render.  Can you write a script to create the array, rather than hard-code it?  And then finish the program.

Sure, here's an updated code example that generates the input array using a loop:

<a href="example-2.py">example 2</a>

In this updated code, we use a loop to generate an **identity matrix** as the **input** `xs` tensor, and an array of **consecutive numbers** as the **output** `ys` tensor.

We then train the model on this data, and finally use another loop to generate an **input tensor of all zeros to make a prediction**, which we print to the console.

Note that because this example uses the `tfjs-node` package, you'll need to run this code using Node.js rather than a browser.

## Tensorflow / Node

To use TensorFlow.js in a Node.js environment, you can install the `@tensorflow/tfjs-node` package, which includes the TensorFlow.js library optimized for use with Node.js.

To use the `tf` variable to reference the TensorFlow.js library in your code, you can import it at the beginning of your script using the `require()` function.

### What the heck is this (1)

Can you explain what this is doing?

```js
const ys = tf.tensor2d(new Array(100).fill(0).map((_, i) => [i]));
```

Certainly!

In this line of code, we're using the `tf.tensor2d()` function to create a new 2D tensor, which will be used as the **output** (`ys`) of our machine learning model.

Here's a breakdown of the code:

* `new Array(100)` creates a new array with 100 elements.
* `.fill(0)` fills each element of the array with the value 0. So now we have an array of 100 zeros.
* `.map((_, i) => [i])` maps each element of the array to a new array containing a single element, which is the index of that element in the original array. The underscore _ is used as a placeholder for the first argument of the `map()` function, which is not needed in this case.

So the end result is an array of 100 sub-arrays, each containing a single number representing the index of that sub-array in the original array. This array is then passed to the `tf.tensor2d()` function, which creates a new 2D tensor with this data.

In other words, we're **creating a simple dataset** where

* the input is a 100-dimensional vector of all zeros, and 
* the output is a 100-dimensional vector
  * where each element corresponds to its index in the vector.

This is a toy example used to demonstrate how to train and use a basic machine learning model with TensorFlow.js. 🚂

### What the heck is this (2)

```js
const xs = tf.tensor2d(new Array(100).fill(0).map((_, i) => {
  return new Array(100).fill(0).map((_, j) => {
    return i === j ? 1 : 0;
  });
}));
```

Here is an alternative way of writing that line of code that might be easier to understand:

<a href="create-100x100.js">create-100x100.js</a>


This code does the same thing as the previous code, but it's broken down into smaller steps to make it easier to understand. Here's what it does:

1. Create an empty array called `rows` to hold the rows of our 2D tensor.
1. Loop 100 times to create 100 rows for our tensor.
1. Create an empty array called `row` to represent one row of the tensor.
1. Loop 100 times to create 100 elements for this row.
1. If we're at the diagonal of the tensor (i.e., if `i === j`), set this element to 1, otherwise set it to 0.
1. Add this element to the `row` array.
1. After we've looped 100 times to create all the elements for this row, add this `row` array to our `rows` array.
1. After we've looped 100 times to create all 100 rows for our tensor, create a 2D tensor using `tf.tensor2d()` with the rows array as the argument.


## What is "diagonal of the tensor" and why is it important

The diagonal of a 2D tensor is the **set of elements** that run from the **top left to the bottom right** corner of the tensor. For example, in a 3x3 tensor, the diagonal would be the elements at (0, 0), (1, 1), and (2, 2).

In the code we looked at earlier, we were creating a 2D tensor where the elements on the diagonal are 1, and all other elements are 0.

This is important, because this particular tensor is known as the **identity matrix**, which is a very useful mathematical tool...

The identity matrix is used to perform certain types of **matrix operations**, such as multiplying a matrix by the identity matrix, which results in the same matrix.

It's also used to **invert** matrices, which is an important operation in many fields.

So even though the code we looked at earlier might seem a bit confusing at first, it's actually creating a very useful mathematical object that has many applications!

## In the browser

Is there a way to use TensorFlow.js in the browser, rather than using node.js via command line?

Yes, you can use TensorFlow.js directly in the browser! You can either download the library and include it in your HTML file using a `<script>` tag, or you can use a content delivery network (CDN) to load the library directly from a remote server.

Here's an example of how you can include TensorFlow.js using a CDN...

In this example, we're including TensorFlow.js from the jsDelivr CDN using the src attribute of the `<script>` tag. You can replace the version number (4.8.0 in this case) with the version you want to use.

After you've included TensorFlow.js in your HTML file, you can write your TensorFlow.js code directly in a `<script>` tag. For example:

<a href="index.html">index.html</a>

This code will create a 2D tensor with some data and log it to the console. You can open this HTML file in your browser to see the results!

## What is the JavaScript equivalent of Python's array.shape?

No problem!  In vanilla JavaScript, you can get the shape of an array using the `.length` property of each dimension:

<a href="get-shape.js">function vanilla</a>

In this example, we're creating a 2D array with some data, getting its shape by getting the `.length` property of each dimension and storing them in an array.

Note that we're assuming that all rows have the same length, so we're getting the `.length` property of the first row (`arr[0].length`) to get the number of columns.

If the rows have different lengths, you'll need to modify this code to handle that case.


## Here's an example of a 3D array

<a href="get-shape.js">function tree\_dee</a>


In this example, we're creating a 3D array with some data.

The array has two "sheets", each of which has three "rows", and each row has two "columns".

We're getting the shape of the array by getting the `.length` property of each dimension and storing them in an array.

Note that we're assuming that all "sheets", "rows", and "columns" have the same length, so we're getting the `.length` property of the first "row" and "column" of the first "sheet" to get the number of "rows" and "columns".

If any of the dimensions have different lengths, you'll need to modify this code to handle that case.

## Here's the adult version

"What The Heck Is This (2)"

Sure, I'd be happy to explain this line of code in a simpler way!

Here's what this line of code is doing:

```javascript
const xs = tf.tensor2d(new Array(100).fill(0).map((_, i) => {
  return new Array(100).fill(0).map((_, j) => {
    return i === j ? 1 : 0;
  });
}));
```

<br>

We're creating a 2D tensor (`xs`) with a shape of `[100, 100]` using the `tf.tensor2d()` function. This tensor will be used as the input to our machine learning model.

Here's a simpler way to break down the code:

1. `new Array(100).fill(0)` **creates a new array** with 100 elements, all set to 0. This will serve as the base array that we'll use to build our 2D tensor.
2. `.map((_, i) => {...})` **maps** each element of the base array to a new value using a function that takes two arguments:
    * The current element (`_`) and 
    * The index of the element (`i`).
3. `new Array(100).fill(0).map((_, j) => {...})` creates a new array of 100 elements, all set to 0. This **nested array** will represent one row of our 2D tensor.
4. `i === j ? 1 : 0` sets the value of each element in the nested array to **either 1 or 0** based on whether the index of the nested array matches the index of the outer array. If the indices match, the element is set to 1, otherwise it's set to 0.
5. The nested array created in step 3 is returned as the new value for the element of the base array. This completes the inner `map()` function, which creates one row of our 2D tensor.
6. The outer `map()` function completes when all elements of the base array have been processed. It returns an **array of 100 nested arrays**, each representing a row of our 2D tensor.
7. The final array of 100 nested arrays is passed to the `tf.tensor2d()` function to create the 2D tensor.

So in simpler terms, we're creating a 2D tensor with a shape of `[100, 100]` where each row is a one-hot encoded vector.

A **one-hot encoded vector** is a vector where only one element is set to 1, and all other elements are set to 0.

In this case, the element that is set to 1 corresponds to the index of the row in the tensor.

For example, the first row will be `[1, 0, 0, ..., 0]`,

The second row will be `[0, 1, 0, ..., 0]`, and so on

Up to the last row, which will be `[0, 0, 0, ..., 1]`

<br>

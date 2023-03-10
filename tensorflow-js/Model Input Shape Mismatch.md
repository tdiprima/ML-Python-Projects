<span style="color:#0000dd;">Hi.  In our tensorflow.js example, we did:</span>

```js
model.fit(xs, ys, {epochs: 100}).then(() => {
  // Make a prediction
  const input = tf.tensor2d(new Array(100).fill(0));
  const output = model.predict(input);
  // Print the output
  output.print();
});
```

<span style="color:#0000dd;">And model.predict(input) did not work, but model.predict(xs) did work.  Any thoughts?</span>

The reason `model.predict(xs)` worked in your TensorFlow.js example but `model.predict(input)` did not work is likely due to a shape mismatch between the **input data** and the model's **input layer.**

In the example you provided, `xs` is likely a tensor of shape **`[num_samples, input_size]`**, where `num_samples` is the number of samples in your input data, and `input_size` is the size of each sample.

**This shape matches the expected input shape** of your model's input layer, so you were able to use `xs` as input for `model.predict()`.

On the other hand, `input` is a tensor of shape `[1, input_size]`, which is a different shape than the expected input shape of the model's input layer. Therefore, when you called `model.predict(input)`, TensorFlow.js likely raised an error indicating a shape mismatch.

To make `model.predict(input)` work, you would need to reshape the input tensor to match the expected input shape of the model's input layer. For example, if the expected input shape of the model's input layer is `[null, input_size]`, you could reshape input to have shape `[1, input_size]` using the `tf.reshape()` method:

```js
const input = tf.tensor2d(new Array(100).fill(0));
input.reshape([1, input_size]);
const output = model.predict(input);
output.print();
```

Note that `input_size` needs to be replaced with the size of each sample in your input data.

<br>

<span style="color:#0000dd;font-size:larger;">tensor2d() requires shape to be provided when values are a flat/TypedArray</span>

<span style="color:#0000dd;font-size:larger;">Really means: add a "shape" argument.</span>

You can specify the shape of a flat/TypedArray input tensor by providing a `shape` argument as the second argument to the `tf.tensor2d()` method:

```js
const input = tf.tensor2d(new Array(100).fill(0), [1, 100]);
const output = model.predict(input);
output.print();
```

In this example, we create a 1x100 tensor by passing a flat array of 100 zeros as the first argument to `tf.tensor2d()`, and passing the shape `[1, 100]` as the second argument.

This creates a tensor with shape `[1, 100]`, which matches the expected input shape of your model's input layer.

<span style="color:#0000dd;">I guess cuz we said:</span>

```js
model.add(
  tf.layers.dense({ 
    units: 1, 
    inputShape: [input_size] 
  })
);
```

Note that the shape argument is optional when passing in a multidimensional array, but is required when passing in a flat array or a TypedArray.

<span style="color:red;">IT WORKED!</span>

<br>

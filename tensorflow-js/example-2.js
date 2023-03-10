// Train a simple model to recognize handwritten digits:
const tf = require('@tensorflow/tfjs-node');

const input_size = 100;

// Define the model
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [input_size] }));

// Compile the model
model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

// Generate an identity matrix as the input tensor
const xs = tf.tensor2d(
  new Array(input_size).fill(0).map((_, i) => {
    return new Array(input_size).fill(0).map((_, j) => {
      return i === j ? 1 : 0;
    });
  })
);

console.log("xs shape:", xs.shape); // [ 100, 100 ]

// Create an array of consecutive numbers as the output tensor
const ys = tf.tensor2d(new Array(input_size).fill(0).map((_, i) => [i]));

console.log("ys shape:", ys.shape); // [ 100, 1 ]

// Train the model
model.fit(xs, ys, { epochs: 100 }).then(() => {
  // Make a prediction
  // Input needs to have shape [null, 100]
  const input = tf.tensor2d(new Array(100).fill(0), [1, 100]);
  const output = model.predict(input);

  // Print the output
  output.print(); // Tensor [[48.5965881],]
});

// Train a simple model to recognize handwritten digits (with error):
const tf = require('@tensorflow/tfjs-node');

// Define the model
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [100] }));

// Compile the model
model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

// Train the model
const xs = tf.tensor2d(
  new Array(100).fill(0).map((_, i) => {
    return new Array(100).fill(0).map((_, j) => {
      return i === j ? 1 : 0;
    });
  })
);

const ys = tf.tensor2d(new Array(100).fill(0).map((_, i) => [i]));

model.fit(xs, ys, { epochs: 100 }).then(() => {
  // Make a prediction
  const input = tf.tensor2d(new Array(100).fill(0));
  const output = model.predict(input);

  // Print the output
  output.print();
});

// Creates a 100x100 identity matrix using TensorFlow.js and prints its shape.
// Generate an identity matrix as the input tensor
const tf = require('@tensorflow/tfjs-node');

// Create an empty array to hold the rows of our 2D tensor
const rows = [];

// Loop over 100 times (for the 100 rows of the tensor)
for (let i = 0; i < 100; i++) {
  // Create an empty array to represent one row of the tensor
  const row = [];
  // Loop over 100 times (for the 100 columns of the tensor)
  for (let j = 0; j < 100; j++) {
    // If we're at the diagonal of the tensor (i === j), set this element to 1,
    // otherwise set it to 0
    if (i === j) {
      row.push(1);
    } else {
      row.push(0);
    }
  }
  // Add the row to our array of rows
  rows.push(row);
}

const shape = [rows.length, rows[0].length];
console.log("shape:", shape);

// Create a tensor from our array of rows
const xs = tf.tensor2d(rows);
console.log("shape:", xs.shape);

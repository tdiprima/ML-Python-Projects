// Demonstrates how to get the shape of 2D and 3D arrays, as well as tensors,
// which echoes Python's array.shape functionality.

function vanilla() {
  // Create a 2D array
  // const arr = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
  const arr = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
  ];

  // Get the shape of the array
  const shape = [arr.length, arr[0].length];

  // Log the shape to the console
  console.log(shape); // Output: [3, 3]
}

vanilla();

function tree_dee() {
  // Create a 3D array, cuz it has 3 levels in it
  const arr = [
    [
      [1, 2],
      [3, 4],
      [5, 6]
    ],
    [
      [7, 8],
      [9, 10],
      [11, 12]
    ]
  ];

  // Get the shape of the array
  const shape = [arr.length, arr[0].length, arr[0][0].length];

  // Log the shape to the console
  console.log(shape); // Output: [2, 3, 2]
}

tree_dee();

function two_dee() {
  const tf = require('@tensorflow/tfjs-node');

  // Create a tensor
  // const input = tf.tensor2d([[1, 2], [3, 4]]);
  const input = tf.tensor2d([
    [1, 2],
    [3, 4]
  ]);

  // Get the shape of the tensor
  const shape = input.shape;

  // Log the shape to the console
  console.log(shape); // Output: [2, 2]
}

// two_dee();

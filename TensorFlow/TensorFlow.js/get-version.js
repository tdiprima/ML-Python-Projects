// Logs the TensorFlow.js version and all information regarding the currently installed TensorFlow.js package.
const tf = require('@tensorflow/tfjs-node');

// Version
console.log("\nVersion:", tf.version.tfjs);

// Everything
console.log("\nEverything:", tf.version);

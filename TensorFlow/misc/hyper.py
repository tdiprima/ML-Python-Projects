"""
loss_fn and optimizer are separate, rather than inline
"""
from timeit import default_timer as timer

import tensorflow as tf

# Define your hyper-parameters
learning_rate = 0.01
num_epochs = 10
batch_size = 32

# Load and preprocess your data
# train_data, train_labels, test_data, test_labels = load_and_preprocess_data()
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.mnist.load_data()

# Define your model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),  # Flatten first.
    tf.keras.layers.Dense(10)
])

# Define your loss function and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

start_time = timer()

# Compile your model
model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=['accuracy'])

# Train your model
history = model.fit(train_data, train_labels,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    validation_data=(test_data, test_labels))

end_time = timer()
total_time = end_time - start_time
print(f"\nTrain time: {total_time:.3f} seconds")
# Train time: 25.725 seconds

# Evaluate your model on test data
test_loss, test_acc = model.evaluate(test_data, test_labels)

# Make predictions with your model
predictions = model.predict(test_data)

# Print out some results
print('Test accuracy:', test_acc)
print('Predictions:', predictions)

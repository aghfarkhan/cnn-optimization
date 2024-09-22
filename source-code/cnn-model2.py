import time
import tensorflow as tf
from tf_keras import layers, models
from sklearn.preprocessing import LabelBinarizer
import tensorflow_model_optimization as tfmot  # Import for pruning

# Load CIFAR-10 dataset (ensure your dataset loading script works here)
from dataset import cifar10
cifar = cifar10.Cifar('./cifar-10-python/cifar-10-batches-py')

# Load CIFAR-10 data
(train_images, train_labels), (test_images, test_labels) = cifar.load_cifar10_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Convert labels to one-hot encoding
lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
test_labels = lb.transform(test_labels)

# Create the CNN model
def build_model():
    model = models.Sequential()
    # First convolutional layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    # Second convolutional layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Third convolutional layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Flatten the output and feed it into a fully connected layer
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))

    # Output layer with 10 units (one for each class) and softmax activation
    model.add(layers.Dense(10, activation='softmax'))
    
    return model

# Wrap the model with prune_low_magnitude for pruning
def apply_pruning_to_model(model):
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.30, final_sparsity=0.80, begin_step=0, end_step=2000
        )
    }
    # Wrapping the entire model for pruning
    model_for_pruning = prune_low_magnitude(model, **pruning_params)
    return model_for_pruning

# Build the model
model = build_model()

# Apply pruning
model = apply_pruning_to_model(model)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Define pruning callbacks to update the pruning process during training
callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir='/tmp/logs')
]

# Measure training time
start_time = time.time()  # Start the timer

# Train the pruned model
model.fit(train_images, train_labels, epochs=10, 
          validation_data=(test_images, test_labels),
          callbacks=callbacks)

# End the timer
end_time = time.time()

# Calculate total training time
total_time = end_time - start_time

# Evaluate the pruned model on the test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print(f'\nTest accuracy: {test_acc}, '
      f'Total training time: {total_time:.2f} seconds')

# Strip the pruning wrappers for final model deployment
model = tfmot.sparsity.keras.strip_pruning(model)

# Save the pruned model
model.save('cifar10_pruned_model.h5')
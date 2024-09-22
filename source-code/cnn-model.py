import time
from keras import layers, models, utils
to_categorical = utils.to_categorical
from sklearn.preprocessing import LabelBinarizer
from dataset import cifar10
# Specify your local directory where CIFAR-10 is saved
cifar = cifar10.Cifar('./cifar-10-python/cifar-10-batches-py')

# Load CIFAR-10 data
(train_images, train_labels), (test_images, test_labels) = cifar.load_cifar10_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

#cifar.show_label_images(train_images, train_labels, 20)

# Convert labels to one-hot encoding
lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
test_labels = lb.transform(test_labels)

# Create the CNN model
model = models.Sequential()

# First convolutional layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# Second convolutional layer
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.MaxPooling2D((2, 2)))

# Third convolutional layer
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Flatten the output and feed it into a fully connected layer
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

# Output layer with 10 units (one for each class) and softmax activation
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Measure training time
start_time = time.time()  # Start the timer

# Train the model
model.fit(train_images, train_labels, epochs=10, 
          validation_data=(test_images, test_labels))

# End the timer
end_time = time.time()

# Calculate total training time
total_time = end_time - start_time

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print(f'\nTest accuracy: {test_acc}, '
      f'Total training time: {total_time:.2f} seconds'
      #f'\n Weights: {model.weights.}'
      '')
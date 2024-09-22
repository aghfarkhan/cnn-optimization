import time
import tensorflow as tf
from keras import layers, models, utils
from sklearn.preprocessing import LabelBinarizer
import tensorflow_model_optimization as tfmot  # For pruning

# Load CIFAR-10 dataset
from dataset import cifar10
cifar = cifar10.Cifar('./cifar-10-python/cifar-10-batches-py')
(train_images, train_labels), (test_images, test_labels) = cifar.load_cifar10_data()

# Normalize the images
train_images, test_images = train_images / 255.0, test_images / 255.0

# Show sample images
cifar.show_label_images(train_images, train_labels, 20)

# One-hot encode the labels
lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
test_labels = lb.transform(test_labels)

# Your original CNN model definition
def build_student_model():
    student_model = models.Sequential()
    student_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    student_model.add(layers.MaxPooling2D((2, 2)))
    student_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    student_model.add(layers.MaxPooling2D((2, 2)))
    student_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    student_model.add(layers.Flatten())
    student_model.add(layers.Dense(64, activation='relu'))
    student_model.add(layers.Dense(10, activation='softmax'))
    return student_model

# Apply pruning to the entire Sequential model
def apply_pruning_to_model(model):
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.30, final_sparsity=0.80, begin_step=0, end_step=2000
        )
    }
    # Wrapping the entire Sequential model for pruning
    model_for_pruning = prune_low_magnitude(model, **pruning_params)
    return model_for_pruning

# Create the student model (your CNN model)
student_model = build_student_model()

# Apply pruning
student_model = apply_pruning_to_model(student_model)

# Compile the student model
student_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create a simple teacher model (your original CNN without pruning for knowledge distillation)
teacher_model = build_student_model()
teacher_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the teacher model first
print("Training Teacher Model...")
teacher_model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Knowledge distillation loss function
def distillation_loss(teacher_logits, student_logits, temperature=5):
    teacher_probs = tf.nn.softmax(teacher_logits / temperature)
    student_probs = tf.nn.softmax(student_logits / temperature)
    return tf.reduce_mean(tf.losses.KLDivergence()(teacher_probs, student_probs))

# Train the student model with distillation
class DistillationModel(tf.keras.Model):
    def __init__(self, student, teacher):
        super(DistillationModel, self).__init__()
        self.student = student
        self.teacher = teacher

    def compile(self, optimizer, metrics, temperature=3):
        super(DistillationModel, self).compile(optimizer=optimizer, metrics=metrics)
        self.temperature = temperature
        self.loss_fn = distillation_loss

    def train_step(self, data):
        x, y = data

        # Forward pass of teacher and student
        teacher_predictions = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            student_predictions = self.student(x, training=True)
            loss = self.loss_fn(teacher_predictions, student_predictions, self.temperature)

        # Backpropagation
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Metrics update
        self.compiled_metrics.update_state(y, student_predictions)
        return {m.name: m.result() for m in self.metrics}

# Create distillation model using the student and teacher
distillation_model = DistillationModel(student=student_model, teacher=teacher_model)
distillation_model.compile(optimizer='adam', metrics=['accuracy'])

# Train the student model using knowledge distillation
print("Training Student Model with Distillation...")
distillation_model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Strip the pruning wrappers and save the pruned model
student_model = tfmot.sparsity.keras.strip_pruning(student_model)
student_model.save('student_model_pruned.h5')

# Quantization with TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(student_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Full Integer Quantization using representative dataset
def representative_dataset_gen():
    for input_value in train_images:
        yield [input_value.reshape(1, 32, 32, 3)]

converter.representative_dataset = representative_dataset_gen
tflite_model = converter.convert()

# Save the quantized TFLite model
with open('student_model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

# Evaluate the student model on test dataset
test_loss, test_acc = student_model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc}')
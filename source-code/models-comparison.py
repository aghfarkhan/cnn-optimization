import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from dataset import cifar10
from tf_keras import models as kerasModels
from tf_keras.utils import to_categorical

# ===== 1. Load CIFAR-10 Dataset =====
cifar = cifar10.Cifar('./cifar-10-python/cifar-10-batches-py')
(train_images, train_labels), (test_images, test_labels) = cifar.load_cifar10_data()

# Keep a copy of original integer labels for TFLite accuracy calculation
original_test_labels = test_labels.copy()

# One-hot encode for teacher/student model evaluation
test_labels = to_categorical(test_labels, num_classes=10)

# ===== 2. Load Saved Models =====
teacher_model = kerasModels.load_model('teacher_model.keras')
student_model = kerasModels.load_model('student_model_pruned.keras')

# ===== 3. Load TFLite Model =====
interpreter = tf.lite.Interpreter(model_path='student_model_quantized.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ===== 4. Measure Inference Time Functions =====
def measure_inference_time(model, images, num_samples=1000):
    start = time.time()
    model.predict(images[:num_samples])
    end = time.time()
    return (end - start) / num_samples  # Average time per sample

def measure_tflite_inference_time(interpreter, images, num_samples=1000):
    start = time.time()
    for i in range(num_samples):
        img = images[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
    end = time.time()
    return (end - start) / num_samples  # Average time per sample

# ===== 5. Compile Models for Evaluation =====
teacher_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
student_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ===== 6. Evaluate Model Accuracy =====
teacher_loss, teacher_acc = teacher_model.evaluate(test_images, test_labels, verbose=0)
student_loss, student_acc = student_model.evaluate(test_images, test_labels, verbose=0)

# ===== 7. Evaluate TFLite Model Accuracy =====
correct_preds = 0
for i in range(len(test_images)):
    img = test_images[i:i+1].astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    
    # Compare predicted class with original (integer) label
    if np.argmax(tflite_output) == original_test_labels[i]:
        correct_preds += 1

student_tflite_acc = correct_preds / len(test_images)

# ===== 8. Get Model Size =====
teacher_params = teacher_model.count_params()
student_params = student_model.count_params()

# ===== 9. Measure Inference Time =====
teacher_time = measure_inference_time(teacher_model, test_images)
student_time = measure_inference_time(student_model, test_images)
student_tflite_time = measure_tflite_inference_time(interpreter, test_images)

# ===== 10. Plot Accuracy Comparison =====
plt.figure(figsize=(10, 5))
plt.bar(['Teacher', 'Student', 'Student (TFLite)'], 
        [teacher_acc, student_acc, student_tflite_acc], 
        color=['blue', 'orange', 'green'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()

# ===== 11. Plot Model Size and Inference Time =====
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(['Teacher', 'Student', 'Student (TFLite)'], 
        [teacher_params, student_params, student_params], 
        color=['blue', 'orange', 'green'])
ax1.set_ylabel('Model Parameters')
ax1.set_title('Model Efficiency Comparison')

ax2 = ax1.twinx()
ax2.plot(['Teacher', 'Student', 'Student (TFLite)'], 
         [teacher_time, student_time, student_tflite_time], 
         color='red', marker='o', label='Inference Time (s/sample)')
ax2.set_ylabel('Inference Time (s/sample)')

plt.legend(loc='upper right')
plt.show()

# ===== 12. Print Results =====
print(f"Teacher Model Accuracy: {teacher_acc:.4f}")
print(f"Student Model Accuracy: {student_acc:.4f}")
print(f"Student TFLite Model Accuracy: {student_tflite_acc:.4f}")

print(f"Teacher Model Parameters: {teacher_params}")
print(f"Student Model Parameters: {student_params}")

print(f"Teacher Inference Time (per sample): {teacher_time:.5f} s")
print(f"Student Inference Time (per sample): {student_time:.5f} s")
print(f"Student TFLite Inference Time (per sample): {student_tflite_time:.5f} s")
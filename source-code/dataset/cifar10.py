import os
import numpy as np
import pickle
from matplotlib import pyplot as plt

class Cifar:
    def __init__(self, dataDirectory):
        self.directory = dataDirectory

    # Function to load the CIFAR-10 dataset from local files
    def load_cifar10_data(self):
        def unpickle(file):
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict
        
        train_images = []
        train_labels = []
        
        # Load training batches
        for i in range(1, 6):
            batch = unpickle(os.path.join(self.directory, f'data_batch_{i}'))
            train_images.append(batch[b'data'])
            train_labels.extend(batch[b'labels'])
        
        train_images = np.concatenate(train_images).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        train_labels = np.array(train_labels)
        
        # Load test batch
        test_batch = unpickle(os.path.join(self.directory, 'test_batch'))
        test_images = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        test_labels = np.array(test_batch[b'labels'])
        
        return (train_images, train_labels), (test_images, test_labels)
    
    def show_label_images(self, train_images, train_labels,num_images=10):
        # Display the first 10 images from the training set
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                    'dog', 'frog', 'horse', 'ship', 'truck']

        plt.figure(figsize=(10,10))
        for i in range(num_images):
            plt.subplot(1,num_images,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i], cmap=plt.cm.binary)
            plt.xlabel(class_names[train_labels[i]])
        plt.show()
    pass
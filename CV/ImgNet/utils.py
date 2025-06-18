""" Utils for importing images and data """

## library imports ##
import os
import cv2
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor


## func to import test and val images
def import_others(img_dir, img_num=int):
    """ Import images from the specified directory """
    start = time.time()
    processed_images = []
    img_paths = []

    def process_image_batches(img_paths, processed_images, img_num):
        """ Import images from the specified directory """

        # read in image, resize, grayscale, and normalize
        for i in range(0, img_num):
            img_path = cv2.imread(img_paths[i])
            img = cv2.resize(img_path, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis = 2)
            img = img / 255.0
            processed_images.append(img)
            completed_percentage = (i / img_num) * 100
            if completed_percentage in [25, 50, 75, 100]:
                print(f"Images processed: {i} ({round(completed_percentage)}%)")

    # collect image paths
    for root, dirs, files in os.walk(img_dir):
            for file in files:
                if file.endswith(".JPEG"):
                     img_paths.append(os.path.join(root, file))

    # shuffle images to get random images from random folders and process while generating labels
    random.shuffle(img_paths)
    process_image_batches(img_paths, processed_images, img_num)

    # return run stats and data        
    end = time.time()
    print(f"Function processed {img_dir} in {round(end - start)} seconds.\n")
    return processed_images


## func to view training images with their assigned labels
def view_train_images(x, y, n = 5):
    """ view train data """
    for img in range(0, 5):
        plt.imshow(x[img], cmap="gray")
        plt.show()
        print(y[img])


def plot_training_results(history):
     """ Plot Accuracy Results """
     plt.figure(figsize=(12, 4))
     plt.subplot(1, 2, 1)
     plt.plot(history.history['accuracy'])
     plt.plot(history.history['val_accuracy'])
     plt.title('Model Accuracy')
     plt.xlabel('Epoch')
     plt.ylabel('Accuracy')
     plt.legend(['Train', 'Val'], loc='best')
     plt.show()

     """ Plot Loss Results """
     plt.subplot(1, 2, 2)
     plt.plot(history.history['loss'])
     plt.plot(history.history['val_loss'])
     plt.title('Model Loss')
     plt.xlabel('Epoch')
     plt.ylabel('Loss')
     plt.legend(['Train', 'Val'], loc='best')
     plt.show()


def fast_import2(img_dir, img_num=int):
    """ Import and preprocess images concurrently from the specified directory """
    start = time.time()
    
    # Helper function to process individual images
    def process_image(img_path):
        """ Read and preprocess a single image, returning the processed image and its label """
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=2)
        img = img / 255.0
        label = img_path.split(os.path.sep)[-2]  # label is the parent folder of the image
        return img, label

    # Collect image paths
    img_paths = [os.path.join(root, file) 
                 for root, dirs, files in os.walk(img_dir) 
                 for file in files if file.endswith(".JPEG")]

    # Shuffle images to get random images from random folders
    random.shuffle(img_paths)
    img_num = min(img_num, len(img_paths))
    img_paths = img_paths[:img_num]

    processed_images = []
    labels = []

    # Use ThreadPoolExecutor to process images concurrently
    with ThreadPoolExecutor(max_workers=8) as executor:  # You can adjust the number of threads here
        results = executor.map(process_image, img_paths)

    # Collect processed images and labels
    for img, label in results:
        processed_images.append(img)
        labels.append(label)

    # Output stats
    end = time.time()
    print(f"Function processed {img_num} images in {round(end - start)} seconds.\n")

    return processed_images, labels


# func to get data for CV project
def get_images(train_dir, val_dir, train_num = 0):
    """ Get data for CV project """
    X_train, Y_train = fast_import2(train_dir, train_num)
    X_val, Y_val = fast_import2(val_dir, int(.15 * train_num))
    return X_train, Y_train, X_val, Y_val
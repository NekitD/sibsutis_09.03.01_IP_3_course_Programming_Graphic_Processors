import pickle
import gzip
import numpy as np
import struct

def save_binary(filename, data):
    with open(filename, 'wb') as f:
        f.write(data.tobytes())

def convert_mnist_pkl():
    with gzip.open('img/mnist.pkl.gz', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    
    train_images, train_labels = training_data
    val_images, val_labels = validation_data
    test_images, test_labels = test_data
    
    save_binary('img/train_images.bin', train_images.astype(np.float32))
    save_binary('img/train_labels.bin', train_labels.astype(np.int32))
    save_binary('img/val_images.bin', val_images.astype(np.float32))
    save_binary('img/val_labels.bin', val_labels.astype(np.int32))
    save_binary('img/test_images.bin', test_images.astype(np.float32))
    save_binary('img/test_labels.bin', test_labels.astype(np.int32))
    
    with open('img/dims.txt', 'w') as f:
        f.write(f"{train_images.shape[0]} {train_images.shape[1]}\n")
        f.write(f"{val_images.shape[0]} {val_images.shape[1]}\n")
        f.write(f"{test_images.shape[0]} {test_images.shape[1]}\n")
    
    print(f"Размер обучающей выборки: {train_images.shape[0]} x {train_images.shape[1]}")
    print(f"Размер валидационной: {val_images.shape[0]} x {val_images.shape[1]}")
    print(f"Размер тестовой: {test_images.shape[0]} x {test_images.shape[1]}")
    print("Конвертация завершена!")

if __name__ == "__main__":
    convert_mnist_pkl()
# KNN Digit Recognizer using MNIST Dataset

## Overview
This project implements the K-Nearest Neighbors (KNN) algorithm to classify handwritten digits from the MNIST dataset. The MNIST dataset is a well-known benchmark in machine learning and consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels.

## Features
- Implementation of KNN from scratch and using `sklearn`
- Data preprocessing and visualization
- Hyperparameter tuning for optimal performance
- Performance evaluation using accuracy metrics
- Comparison with other classifiers

## Dataset
The MNIST dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/avnishnish/mnist-original) or directly loaded via `tensorflow.keras.datasets.mnist`.

## Installation
To run this project, ensure you have the following dependencies installed:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

## Usage
Run the following script to train and evaluate the model:

```bash
python knn_mnist.py
```

## Implementation Details
1. Load the MNIST dataset and preprocess the data.
2. Flatten the images from 28x28 to a 1D array of size 784.
3. Split the dataset into training and testing sets.
4. Implement KNN using Euclidean distance.
5. Optimize the value of K using cross-validation.
6. Evaluate model performance using accuracy scores.

## Code Explanation
### Load and Preprocess Data
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('mnist_train.csv')
data = df.values
x = data[:, 1:]
y = data[:, 0]

split = int(0.8 * x.shape[0])
x_train = x[:split, :]
y_train = y[:split]
x_test = x[split:, :]
y_test = y[split:]
```

### Visualize Sample Image
```python
def drawimg(sample):
    img = sample.reshape((28, 28))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

drawimg(x_train[1])
print(y_train[1])
```

### KNN Algorithm Implementation
```python
def dist(x1, x2):
    return np.sqrt(sum((x1 - x2) ** 2))

def knn(X, Y, queryPoint, k=5):
    vals = []
    m = X.shape[0]
    
    for i in range(m):
        d = dist(queryPoint, X[i])
        vals.append((d, Y[i]))
    
    vals = sorted(vals)[:k]
    vals = np.array(vals)
    new_vals = np.unique(vals[:, 1], return_counts=True)
    index = new_vals[1].argmax()
    pred = new_vals[0][index]
    
    return pred

pred = knn(x_train, y_train, x_test[1])
print(int(pred))
```

## Results
- The model achieves an accuracy of ~96% on the test dataset with an optimized K value.
- Performance comparison with other ML models such as SVM, Random Forest, and Neural Networks.

## Contributions
Feel free to contribute by submitting issues or pull requests.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author
**Akshat Bhatnagar** - *Future of Data Science* 🚀


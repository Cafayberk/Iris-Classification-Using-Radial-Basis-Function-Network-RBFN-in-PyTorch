# Iris-Classification-Using-Radial-Basis-Function-Network-RBFN-in-PyTorch

This project implements a **Radial Basis Function Network (RBFN)** to classify the famous **Iris dataset** using **PyTorch**.  
The model is trained and tested in **Google Colab** with data loaded directly from **Google Drive**.

---

## Project Overview
- **Framework**: PyTorch
- **Dataset**: Iris dataset (`iris.data`)
- **Environment**: Google Colab
- **Key Steps**:
  1. Load dataset from Google Drive
  2. Preprocess and standardize data
  3. Train an RBFN model
  4. Evaluate accuracy on test data

---

## Dataset
The dataset used is the **Iris dataset** (`iris.data`) stored in Google Drive.  
File path used in the project: /content/drive/MyDrive/iris/iris.data


You can download the original dataset from:  
[UCI Machine Learning Repository - Iris](https://archive.ics.uci.edu/dataset/53/iris)

---

## ⚙️ How to Run in Google Colab

### 1. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
After mounting, your files will be accessible in /content/drive/MyDrive/.

2. Install Required Libraries (if not already installed)
!pip install torch torchvision torchaudio scikit-learn pandas

3. Import the Dataset from Google Drive

Make sure the dataset is stored in: /content/drive/MyDrive/iris/iris.data

Read the dataset in your code:
import pandas as pd
file_path = '/content/drive/MyDrive/iris/iris.data'
df = pd.read_csv(file_path, header=None)

Model Description

Input Layer: 4 features (sepal length, sepal width, petal length, petal width)

Hidden Layer: RBF layer with Gaussian kernels

Output Layer: 3 neurons (Setosa, Versicolor, Virginica)

The RBFN architecture:

Convert inputs to RBF activations using Gaussian kernel

Pass activations to a fully connected layer

Use CrossEntropyLoss for training

Optimize with Adam optimizer

Training & Accuracy

Epochs: 100

Optimizer: Adam (lr=0.01)

Loss Function: CrossEntropyLoss

Final accuracy will be printed after training.

Example output:
Epoch [10/100], Loss: 0.9876
...
Accuracy: 96.67%

How to Use

Open the .ipynb file in Google Colab

Mount Google Drive

Ensure the dataset path matches your Drive folder

Run all cells to train and evaluate the model

This project is released under the MIT License.

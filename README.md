# Handwritten Digit Recognition using Artificial Neural Networks (ANN)

**Author:** Rudhra Sitholey  
**Registration Number:** 23BCY10296  
**Assignment:** AI-ML Assignment – 8  
**GitHub Repository:** [Handwritten-Digit-Recognition-using-Artificial-Neural-Networks--ANN-](https://github.com/rudhra23bcy10296/Handwritten-Digit-Recognition-using-Artificial-Neural-Networks--ANN-)

---

## 📌 Objective

The objective of this project is to develop an **Artificial Neural Network (ANN)** model to classify handwritten digits (0–9) using the MNIST dataset. Automated handwritten digit recognition is essential for postal service organizations to automate zip code reading and streamline mail processing pipelines.

---

## 🔗 Dataset Link

- **Dataset Name:** MNIST Handwritten Digits Dataset (in CSV format)
- **Kaggle URL:** [MNIST in CSV on Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
- **Dataset Description:** 
  - `mnist_train.csv`: 60,000 samples
  - `mnist_test.csv`: 10,000 samples
  - **Features:** 784 pixel values (`1x1` to `28x28`), ranging from 0 to 255 (grayscale intensity)
  - **Target Variable:** `label` (digits 0 to 9)

---

## 🛠️ Libraries Used

- **Deep Learning Framework:** `TensorFlow`, `Keras` (`Sequential`, `Dense`, `Input`, `to_categorical`)
- **Data Manipulation & Analysis:** `Pandas`, `NumPy`
- **Model Evaluation & Preprocessing:** `scikit-learn` (`train_test_split`, `confusion_matrix`, `classification_report`)
- **Visualization:** `Matplotlib`, `Seaborn`

---

## 🔬 Methodology

1. **Data Understanding & Inspection:**
   - Loaded training and testing CSV datasets into Pandas.
   - Identified input features (784 pixel columns) and target label (`label`).
   - Inspected feature distributions and visualized sample handwritten digits using Matplotlib (`plt.imshow`).

2. **Data Preprocessing:**
   - Verified zero missing values across all rows and columns.
   - Separated feature inputs ($X$) and target labels ($y$).
   - Normalized raw pixel intensities from $[0, 255]$ to range $[0, 1]$ via float division ($X / 255.0$).
   - Split dataset into an **80% training set** (56,000 samples) and a **20% testing set** (14,000 samples).
   - One-hot encoded integer labels (0–9) into 10-dimensional categorical vectors using `to_categorical`.

3. **Model Development & Architecture:**
   - Constructed a Multi-Layer Perceptron / Feedforward ANN using Keras `Sequential` API.
   - Compiled the network with **Adam** optimizer, **Categorical Crossentropy** loss function, and **Accuracy** metric.
   - Trained the model for **10 epochs** with a batch size of 64.

4. **Model Evaluation:**
   - Evaluated test loss and accuracy on unseen test data.
   - Generated classification report (Precision, Recall, F1-Score per digit).
   - Visualized confusion matrix heatmap to analyze misclassification patterns.
   - Plotted **Accuracy vs. Epoch** and **Loss vs. Epoch** curves.

---

## 🏗️ Model Architecture

| Layer | Type | Output Shape | Activation | Number of Neurons | Parameters |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Input** | InputLayer | `(None, 784)` | None | 784 | 0 |
| **Hidden Layer 1** | Dense | `(None, 128)` | ReLU | 128 | 100,480 |
| **Hidden Layer 2** | Dense | `(None, 64)` | ReLU | 64 | 8,256 |
| **Output Layer** | Dense | `(None, 10)` | Softmax | 10 | 650 |

- **Total Trainable Parameters:** 109,386

---

## 📊 Results & Performance

- **Test Accuracy:** **> 97.5%**
- **Test Categorical Loss:** **< 0.09**
- **Per-Digit Metrics:** High Precision, Recall, and F1-Scores exceeding **0.97** across all digits (0–9).

### Performance Graphs
- **Accuracy vs Epoch:** Shows rapid increase in training and validation accuracy over the 10 epochs.
- **Loss vs Epoch:** Shows steady, monotonic reduction in loss without signs of overfitting.

---

## 🎯 Conclusion

### Key Findings & Summary (100–150 Words)

This project successfully built and trained a 3-layer **Artificial Neural Network (ANN)** using TensorFlow/Keras for classifying handwritten digits from the MNIST dataset. Through pixel normalization ($[0,1]$ scaling) and one-hot encoding, the ANN model achieved over **97.5% classification accuracy** on unseen test data, demonstrating robust generalization for automated postal code recognition.

**Importance of Hidden Layers:** Hidden layers with non-linear **ReLU** activations allow the neural network to capture intricate, non-linear relationships and hierarchical visual patterns from raw pixel inputs.

**Deep Learning vs Traditional ML:** Deep Learning automatically learns relevant feature representations directly from raw data pixels, eliminating manual feature extraction steps required by traditional ML algorithms.

**Limitation of ANN:** ANNs demand significant computational power, large training sets to avoid overfitting, and act as complex "black box" models with low intrinsic interpretability.

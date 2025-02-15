# 🚀 Bank Card Fraud Detection

## 📌 Project Overview
I developed a **bank card fraud detection system** using **machine learning and deep learning** models. The goal was to detect fraudulent transactions based on past transaction data. The dataset used was the **Credit Card Fraud Detection dataset** from Kaggle.

## 🗂 Dataset & Preprocessing
- **Dataset:** Used `creditcard.csv` containing transaction records.
- **Preprocessing Steps:**
  - Standardized `Amount` column using `RobustScaler` to handle outliers.
  - Normalized `Time` column using Min-Max scaling.
  - Split the dataset into **training (240,000), testing (22,000), and validation (remaining)**.

## 🔍 Exploratory Data Analysis (EDA)
- Plotted **histograms** to visualize feature distributions.
- Checked **class imbalance** (fraud vs. non-fraud transactions).
- Verified correlation between features.

## 🤖 Machine Learning Models Implemented
### **1️⃣ Logistic Regression**
- Implemented `LogisticRegression()` from `sklearn`.
- Evaluated accuracy on training and validation data.

### **2️⃣ Random Forest Classifier**
- Used `RandomForestClassifier(max_depth=2, n_jobs=-1)`.
- Reduced overfitting by limiting tree depth.
- Evaluated with precision, recall, and F1-score.

### **3️⃣ Gradient Boosting Classifier**
- Implemented `GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1)`.
- Used an **adaptive learning approach** for better fraud detection.

### **4️⃣ Support Vector Machine (SVM)**
- Used `LinearSVC(class_weight='balanced')` to handle class imbalance.
- Optimized for high-dimensional feature space.

## 🧠 Deep Learning Model
### **Shallow Neural Network**
- Built a simple **3-layer neural network** using TensorFlow/Keras.
- Used:
  - `Dense` layers with `ReLU` activation.
  - `BatchNormalization()` for stable training.
  - `sigmoid` activation for fraud classification.
- Trained for **10 epochs** with `ModelCheckpoint()` to save the best model.

## 📊 Model Evaluation
- Used **classification reports** to analyze precision, recall, and F1-score.
- Compared different models:
  - **Random Forest** and **Gradient Boosting** performed well on imbalanced data.
  - **Neural Network** achieved high accuracy but required further tuning.
  - **SVM** handled class imbalance effectively.

## 🏆 Key Takeaways
- **Balancing the dataset** is crucial for fraud detection models.
- **Random Forest and Gradient Boosting** were the best-performing models.
- **Neural Networks** need more fine-tuning but show potential.
- **Feature scaling (RobustScaler, Min-Max Scaling)** improves model stability.

## 🔥 Next Steps
- Fine-tune hyperparameters for better fraud detection.
- Implement **oversampling (SMOTE)** to handle imbalanced data.
- Deploy the best model into a **real-time fraud detection system**.

---
👨‍💻 **Developed by:** [Yesbol Yerlan]  

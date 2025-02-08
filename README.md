# AI-Based Disease Prediction System  

## 📌 Overview  
This project is an **AI-powered disease prediction system** that utilizes machine learning models to predict **diabetes, heart disease, and Parkinson’s disease** based on user-provided medical data. The system provides predictions through a **Streamlit-based web UI** and allows users to check their health status using different models.  

## 🏥 Diseases Covered  
1. **Diabetes Prediction** – Uses medical indicators like glucose level, BMI, insulin, and more.  
2. **Heart Disease Prediction** – Analyzes blood pressure, cholesterol levels, and other cardiovascular risk factors.  
3. **Parkinson’s Disease Prediction** – Uses vocal and motor function data to identify early signs.  

## 🔍 Machine Learning Models Used  
We implemented and compared multiple ML models to find the most accurate one:  
- **Random Forest Classifier**  
- **Logistic Regression**  
- **Decision Tree Classifier**  
- **Linear Regression (for understanding trends, not classification)**  

We evaluated these models using **accuracy, precision, recall, and F1-score** to determine the best-performing algorithm.  

## 🐂 Dataset and Preprocessing  
- **Data Used:** Publicly available datasets from medical research studies for diabetes, heart disease, and Parkinson’s disease.  
- **Class Imbalance Handling:** We used **Tomek Links undersampling** to balance the dataset by removing ambiguous samples that are too close to each other. This helps improve the model's ability to generalize.  

## 🎯 Why Accuracy, Precision, and Recall Matter  
When evaluating a medical prediction model, we don’t just rely on **accuracy** because:  
- **Precision:** Measures how many of the predicted positive cases were actually positive. (Important in cases where false positives must be minimized)  
- **Recall (Sensitivity):** Measures how many actual positive cases were correctly identified. (Important when false negatives must be minimized)  
- **F1-Score:** A balance between precision and recall.  

For example, in **diabetes prediction**, missing an actual case (false negative) can be **dangerous**, so **recall is more important**. In contrast, for **heart disease**, a high false positive rate could lead to unnecessary anxiety and tests, so **precision should also be high**.  

## 🌊 Model Evaluation Metrics  
| Model | Accuracy | Precision | Recall | F1-Score |  
|--------|----------|-----------|--------|---------|  
| **Random Forest** | 80.4% | 71.7% | 68.7% | 70.1% |  
| **Logistic Regression** | 78.9% | 69.5% | 66.2% | 67.8% |  
| **Decision Tree** | 76.2% | 67.8% | 64.9% | 66.3% |  

Random Forest performed best overall, but logistic regression was close behind.  

## 🖥️ Web UI Implementation  
- We built an interactive **Streamlit** web application that allows users to enter medical details and get instant predictions.  
- If any field is left empty or contains a default value (like 0 for numerical inputs), the app warns the user before making a prediction.  

## 🛠️ How to Improve Model Performance  
To **increase accuracy, precision, and recall**, we can:  
1. **Use Ensemble Learning** – Combining Random Forest, Decision Trees, and Logistic Regression can reduce model bias and variance.  
2. **Feature Selection** – Removing irrelevant or highly correlated features can make the model more effective.  
3. **Hyperparameter Tuning** – Adjusting parameters like tree depth (Decision Tree), number of estimators (Random Forest), and regularization (Logistic Regression) can optimize performance.  
4. **Prevent Overfitting** – Using **cross-validation**, **pruning (Decision Tree)**, and **regularization (L1/L2 in Logistic Regression)** ensures the model generalizes well.  
5. **Handling Imbalanced Data** – Applying Tomek Links, SMOTE (Synthetic Minority Over-sampling Technique), or weighted loss functions to improve recall for underrepresented classes.  

## 📈 Preventing Overfitting – A Story for Graduate Students & Professors  
Imagine you're a **medical researcher** developing an AI system for early disease detection. You train a model and get **98% accuracy**, but when tested on real patients, it performs **poorly**. What went wrong?  

1. **Overfitting**: The model memorized the training data instead of learning general patterns.  
2. **High Variance**: It performs well on known cases but fails on new ones.  
3. **Solution?** We **apply Tomek Links undersampling**, tune hyperparameters, and use **cross-validation** to ensure it learns real medical insights, not just training set quirks.  

The result? A robust model that **generalizes well** and truly helps doctors and patients.  

---  

## 🚀 Getting Started  
### Installation  
```bash  
pip install -r requirements.txt  
```  
### Run the App  
```bash  
streamlit run web.py  
```  

## 📌 Conclusion  
This project showcases how **machine learning can assist in disease prediction**. By comparing models, handling class imbalance, and designing a user-friendly UI, we provide a practical AI tool for early diagnosis. Future improvements include **deep learning models** and **real-time data integration** from wearables.  

---


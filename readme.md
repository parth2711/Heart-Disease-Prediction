**Project Overview**

This project implements an end-to-end Machine Learning workflow to predict the presence of heart disease using the k-Nearest Neighbors (kNN) algorithm.
The goal of this project was not only to build a working model, but to understand how distance-based models behave, how preprocessing affects predictions, and how models can be deployed as interactive web applications using Streamlit.

**Objectives**

1\. Build a complete ML pipeline from data loading to deployment
2\. Understand kNN behavior and its limitations
3\. Apply feature scaling for distance-based models
4\. Deploy the trained model using Streamlit

**Machine Learning Workflow**

*Data Preparation*

* Used the Heart Disease dataset (heart_disease_uci.csv)
* Converted feature values into numerical data
* Separated features and target variable
* Applied StandardScaler to normalize feature values

*Model Training*

* Trained a k-Nearest Neighbors (kNN) classifier
* Experimented with different values of k
* Saved the trained model and scaler using pickle

*Model Evaluation*

* Confusion Matrix
* Classification Report
* Accuracy Score

**Key Observations & Learnings**

kNN is highly sensitive to feature scaling, making normalization essential. Prediction probabilities from kNN tend to be extreme (0% or 100%) when nearest neighbors strongly agree. This behavior is expected, especially with small k values or clustered data. kNN is effective for learning decision boundaries but is not ideal for calibrated probability estimates. Slight changes in input values can significantly change predictions due to the distance-based nature of the algorithm.

**Deployment**

* The trained model was deployed using Streamlit

* Users can input patient details and receive:

    A binary prediction (Heart Disease / No Heart Disease)

    Prediction probabilities

* The app updates predictions automatically based on user input



*Tech Stack*

* Python
* Pandas
* NumPy
* scikit-learn
* Streamlit


***This project is for educational purposes only and should not be used for real medical diagnosis.***



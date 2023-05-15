# Securing-Finances-Credit-Card-Fraud-Detection

# Our Objective:
To develop and maintain ML-based fraud detection models that are effective at identifying evolving fraud patterns even in the presence of imbalanced data.

# Idea:
We have tested multiple machine learning models on a widely accepted dataset from Kaggle, and our findings have revealed that the Random Forest Classifier outperforms all other models, achieving the highest level of performance metrics.

# Link for dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# Implementation:
Our workflow is as follows:- 
1.	_Data collection_: We have obtained the dataset from Kaggle which contains 0.172 % fraudulent and remaining non-fraudulent transactions.
2.	_Data exploration and EDA_: We have conducted exploratory data analysis to gain insights into the data, check for missing values, and perform feature engineering.
3.	_Data preprocessing_: We have preprocessed the data by normalizing or standardizing it and splitting it into training and testing sets.
4.	_Model selection and training_: We have selected the appropriate machine learning model(s) to detect fraud. The models on which we have trained our models are Random forest classifier, Logistic Regression, Decision Tree, Ensemble Learning, Deep Neural Network & CNN + LSTM.
5.	_Model evaluation and comparing performance metrics_: We have evaluated the trained models on the testing set and analyzed their performance metrics such as accuracy, AUC score, confusion matrix, and classification report.

# Technology Stack:- 
•	_Platform_: Google Colab
•	_Languages_: Python
•	_Libraries Used_:  
   1.	pandas
   2.	numpy
   3.	seaborn
   4.	matplotlib.pyplot
   5.	scikit-learn
   6.	keras

# Applications:
1.	_Predictive modeling_: Use the trained random forest classifier to predict whether a new transaction is fraudulent or not. The model will consider the extracted features as input and output a binary classification result indicating whether the transaction is fraudulent or not.
2.	_Real-time monitoring_: Integrate the predictive model into a real-time monitoring system that analyzes incoming transactions. The system will continuously monitor incoming transactions, and when a new transaction is received, it will apply the random forest model to it to predict whether it is fraudulent or not.
3.	_Alerts_: If the model detects a transaction as fraudulent, the real-time monitoring system can immediately alert the cardholder and/or the financial institution to take appropriate action, such as declining the transaction, blocking the card, or contacting the cardholder to confirm the transaction.
4.	_Continuous learning_: The random forest model can be continuously updated by retraining it on new data and incorporating feedback from fraud analysts. This will help the model adapt to evolving fraud patterns and maintain its effectiveness over time.

# Result:
It can be seen that Random Forest Classifier outperforms all the other ML models:
1.	It can be seen that the Random Forest classifier and Ensemble Learning have the highest accuracy of 100%. Other ML models have an accuracy of 99%.
	_Hence, accuracy is not the only parameter to evaluate the model._
2.	The precision of the Random Forest Classifier is the highest. Ensemble Learning also has a precision with a difference of 1%.
3.	The recall of decision tree and deep learning is the highest.
4.	The F1 score of the random forest classifier and ensemble learning is highest, however, the time taken by the random forest classifier is less than ensemble learning.

# How the idea can be scaled for larger and more advanced problems?
1. 	_Feature engineering_: We need to extract meaningful features from the credit card transaction data such as transaction amount, location, time of day, cardholder's spending patterns, and so on.
2. 	_Real-time monitoring_: Once the model is trained, we can deploy it to a real-time monitoring system that analyzes incoming credit card transactions. If the model detects a transaction as fraudulent, it can immediately alert the cardholder and/or the financial institution to take appropriate action.
3.  _Continuous learning_: To improve the accuracy of the model over time, we can also incorporate continuous learning techniques. This involves retraining the model on new data and incorporating feedback from fraud analysts to refine the model's predictions.


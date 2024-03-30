#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[20]:


import pandas as pd

# Load the dataset
data = pd.read_csv('airlines_reviews.csv')

# Display the first few rows of the dataset to understand its structure
#data.head()


# # Data Analysis
# 
# Title: Nominal - This feature is text-based and unique for most entries. Text preprocessing could be considered if you plan to use it for analysis, such as tokenization or embedding, but it might be more useful for NLP tasks.
# 
# Name: Nominal - Contains names of the reviewers. This feature is not useful for quantitative analysis and could be dropped unless you're interested in identifying repeat reviewers.
# 
# Review Date: Ordinal - Dates can be converted to a datetime format and potentially used to generate time-based features, such as the month or year of the review.
# 
# Airline: Nominal - Categorical data that could be encoded using one-hot encoding or label encoding for model input.
# 
# Verified: Binary - This is a binary feature indicating whether the review is verified. It can be encoded to 0s and 1s (e.g., True to 1, False to 0).
# 
# Reviews: Nominal - This is textual data that could be preprocessed for sentiment analysis or other NLP-based tasks, but not directly useful for quantitative analysis without significant text preprocessing.
# 
# Type of Traveller: Nominal - Categorical data that could be encoded using one-hot or label encoding.
# 
# Month Flown: Ordinal - Could be transformed into numerical format (1-12) or encoded using one-hot encoding if used as a categorical variable.
# 
# Route: Nominal - Contains various flight routes. Could be complex to use directly due to high variability but might be split or encoded for specific analyses.
# 
# Class: Nominal - Categorical data indicating the class of the flight. It should be encoded for model input.
# 
# Seat Comfort, Staff Service, Food & Beverages, Inflight Entertainment, Value For Money, Overall Rating: Ordinal - These are ratings on a scale (likely 1-5 for most, with Overall Rating up to 10), which are already numerical and could be used as-is or normalized.
# 
# Recommended: Binary - Similar to the Verified feature, it can be encoded to 0s and 1s (e.g., Yes to 1, No to 0).

# In[ ]:





# In[21]:


missing_data = data.isnull().sum()
missing_data_summary = missing_data[missing_data > 0]
#missing_data_summary
missing_values = data.select_dtypes(include=['int', 'float']).isnull().sum()
#missing_values


# In[ ]:





# In[22]:


# Encoding the 'Recommended' column
data['Recommended'] = data['Recommended'].map({'yes': 1, 'no': 0})

# Display the transformed data
#data[['Overall Rating', 'Recommended']].head()


# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Features and Target Variable - Adjust according to preprocessing needs
X = data.drop(['Title', 'Name', 'Review Date', 'Reviews', 'Recommended'], axis=1)
y = data['Recommended']

# Preprocessing for numerical and categorical data
numerical_cols = ['Seat Comfort', 'Staff Service', 'Food & Beverages', 'Inflight Entertainment', 'Value For Money', 'Overall Rating']
categorical_cols = [col for col in X.columns if col not in numerical_cols]

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying preprocessing
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Model initialization and training
models = {
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(),
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier()
}

for name, model in models.items():
    print(f"\n{30*'='}\n{name}\n{30*'='}")
    
    # For the Naive Bayes model, convert the sparse matrix to a dense one
    if name == 'Naive Bayes':
        model.fit(X_train_transformed.toarray(), y_train)
        y_pred = model.predict(X_test_transformed.toarray())
    else:
        model.fit(X_train_transformed, y_train)
        y_pred = model.predict(X_test_transformed)
    
    # Evaluation
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


# In[ ]:





import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv('airlines_reviews.csv')

# Selecting only one feature and the target variable
X = data[['Overall Rating']]  # Selecting only 'Overall Rating' as the feature
y = data['Recommended']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model initialization and training (using KNeighborsClassifier)
model = KNeighborsClassifier()

# Fit the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

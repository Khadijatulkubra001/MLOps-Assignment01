import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
data = pd.read_csv('airlines_reviews.csv')

# Encoding the 'Recommended' column
data['Recommended'] = data['Recommended'].map({'yes': 1, 'no': 0})

# Features and Target Variable - Adjust according to preprocessing needs
X = data.drop(['Title', 'Name', 'Review Date', 'Reviews', 'Recommended'], axis=1)
y = data['Recommended']

# Preprocessing for numerical and categorical data
numerical_cols = ['Seat Comfort', 'Staff Service', 'Food & Beverages', 
                  'Inflight Entertainment', 'Value For Money', 'Overall Rating']
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

# Model initialization and training (using KNeighborsClassifier)
model = KNeighborsClassifier()

# Fit the model
model.fit(X_train_transformed, y_train)

# Predictions
y_pred = model.predict(X_test_transformed)

# Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

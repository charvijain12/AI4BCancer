# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the Wisconsin Breast Cancer dataset
file_path = r'C:\Users\mkg_g\Downloads\data.csv'
df = pd.read_csv(file_path)

# Drop unnecessary columns
df = df.drop(columns=['id', 'Unnamed: 32'])

# Encode the 'diagnosis' column (M=1, B=0)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])

# Handle missing values by dropping rows with null values
df = df.dropna()

# Split the dataset into features (X) and target variable (y)
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the base classifier (Decision Tree in this case)
base_classifier = DecisionTreeClassifier(max_depth=1)  # You can change max_depth as needed

# Initialize the AdaBoost classifier
adaboost_classifier = AdaBoostClassifier(base_classifier, n_estimators=100, random_state=42)

# Train the AdaBoost classifier
adaboost_classifier.fit(X_train_scaled, y_train)

# Make predictions
predictions = adaboost_classifier.predict(X_test_scaled)

# Print accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.4f}')

# Print classification report
class_report = classification_report(y_test, predictions)
print('Classification Report:')
print(class_report)
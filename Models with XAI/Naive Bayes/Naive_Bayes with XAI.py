#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from lime.lime_tabular import LimeTabularExplainer
import numpy as np


# In[2]:


# Load the Wisconsin Breast Cancer dataset
file_path = r"C:\Users\Shaurya\Desktop\AI4BCancer\Dataset\data.csv"
df = pd.read_csv(file_path)


# In[3]:


# Drop unnecessary columns
df = df.drop(columns=['id', 'Unnamed: 32'])


# In[4]:


# Encode the 'diagnosis' column (M=1, B=0)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])


# In[5]:


# Handle missing values by dropping rows with null values
df = df.dropna()


# In[6]:


# Split the dataset into features (X) and target variable (y)
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']


# In[7]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[9]:


# Implement Gaussian Naive Bayes
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train_scaled, y_train)


# In[10]:


# Evaluate the model on the test set
y_pred = naive_bayes_model.predict(X_test_scaled)


# In[11]:


# Print accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)


# In[12]:


# Create a LimeTabularExplainer
explainer = LimeTabularExplainer(
    training_data=X_train_scaled,
    mode="classification",
    training_labels=y_train,
    feature_names=X.columns,
    class_names=['Benign', 'Malignant']
)

# Choose a specific instance to explain
instance_index = 0  # Change this to the index of the instance you want to explain
exp = explainer.explain_instance(X_test_scaled[instance_index], naive_bayes_model.predict_proba, num_features=len(X.columns))

# Display the explanation
exp.show_in_notebook(show_table=True)


# In[13]:


# Extracting feature contributions for the specific instance
feature_contributions = exp.as_list()


# In[14]:


# Displaying feature contributions
print("Feature Contributions for the specific instance:")
for feature, contribution in feature_contributions:
    print(f"{feature}: {contribution}")


# In[17]:


# Detailed explanations for each feature based on the LIME output
feature_explanations = {
    "-0.64 < area_worst <= -0.34": "Larger values of 'area_worst' within this range contribute negatively to the prediction, suggesting that smaller worst areas are associated with a lower likelihood of malignancy.",
    "-0.66 < area_mean <= -0.29": "Higher values of 'area_mean' within this range have a negative contribution, indicating that smaller mean tumor areas are associated with a lower risk of malignancy.",
    "-0.69 < perimeter_worst <= -0.28": "Smaller values of 'perimeter_worst' in this range negatively impact the prediction, suggesting that tumors with smaller worst perimeters are less likely to be malignant.",
    "-0.69 < perimeter_mean <= -0.24": "Smaller values of 'perimeter_mean' within this range have a negative contribution, indicating that tumors with smaller mean perimeters are associated with a lower risk of malignancy.",
    "-0.33 < area_se <= 0.08": "Smaller values of 'area_se' in this range negatively affect the prediction, suggesting that tumors with less variability in their areas are less likely to be malignant.",
    "-0.28 < perimeter_se <= 0.20": "Smaller values of 'perimeter_se' within this range contribute negatively, indicating that tumors with less variability in their perimeters are less likely to be malignant.",
    "-0.28 < radius_se <= 0.23": "Smaller values of 'radius_se' within this range negatively impact the prediction, suggesting that tumors with less variability in their radii are less likely to be malignant.",
    "-0.68 < radius_mean <= -0.23": "Smaller values of 'radius_mean' in this range have a negative contribution, indicating that tumors with smaller mean radii are associated with a lower risk of malignancy.",
    "-0.66 < radius_worst <= -0.26": "Smaller values of 'radius_worst' within this range contribute negatively, suggesting that smaller worst radii are associated with a lower likelihood of malignancy.",
    "-0.21 < concavity_se <= 0.30": "Higher values of 'concavity_se' within this range have a positive contribution, indicating that tumors with more pronounced concave portions are more likely to be malignant.",
    "-0.39 < concave points_mean <= 0.67": "Smaller values of 'concave points_mean' within this range negatively impact the prediction, suggesting that tumors with fewer concave points in their contours are less likely to be malignant.",
    "-0.71 < texture_mean <= -0.12": "Lower values of 'texture_mean' in this range have a negative contribution, indicating that tumors with less variation in texture are less likely to be malignant.",
    "-0.28 < compactness_worst <= 0.57": "Smaller values of 'compactness_worst' within this range contribute negatively, suggesting that tumors with less variability in worst compactness are less likely to be malignant.",
    "-0.69 < compactness_se <= -0.28": "Higher values of 'compactness_se' within this range have a positive contribution, indicating that tumors with more variability in compactness are more likely to be malignant.",
    "-0.07 < symmetry_mean <= 0.54": "Lower values of 'symmetry_mean' in this range have a negative contribution, indicating that tumors with less variation in symmetry are less likely to be malignant.",
    "-0.34 < concavity_mean <= 0.55": "Higher values of 'concavity_mean' within this range have a positive contribution, suggesting that tumors with more pronounced concavity are more likely to be malignant.",
    "-0.24 < compactness_mean <= 0.53": "Smaller values of 'compactness_mean' within this range contribute negatively, indicating that tumors with less variability in mean compactness are less likely to be malignant.",
    "-0.21 < fractal_dimension_worst <= 0.46": "Smaller values of 'fractal_dimension_worst' within this range have a negative contribution, indicating that tumors with less complexity in worst fractal dimension are less likely to be malignant.",
    "-0.03 < smoothness_worst <= 0.63": "Higher values of 'smoothness_worst' within this range have a positive contribution, indicating that tumors with smoother worst texture are more likely to be malignant.",
    "-0.68 < texture_se <= -0.20": "Higher values of 'texture_se' within this range have a positive contribution, indicating that tumors with more variation in texture are more likely to be malignant.",
    "-0.18 < fractal_dimension_mean <= 0.46": "Higher values of 'fractal_dimension_mean' within this range have a positive contribution, indicating that tumors with more complexity in mean fractal dimension are more likely to be malignant.",
    "-0.22 < fractal_dimension_se <= 0.25": "Smaller values of 'fractal_dimension_se' within this range have a negative contribution, indicating that tumors with less variability in fractal dimension are less likely to be malignant.",
    "-0.08 < smoothness_mean <= 0.63": "Higher values of 'smoothness_mean' within this range have a positive contribution, indicating that tumors with smoother mean texture are more likely to be malignant.",
    "-0.74 < texture_worst <= -0.05": "Higher values of 'texture_worst' within this range have a positive contribution, indicating that tumors with more variation in worst texture are more likely to be malignant.",
    "-0.12 < symmetry_worst <= 0.43": "Higher values of 'symmetry_worst' within this range have a positive contribution, indicating that tumors with more variation in worst symmetry are more likely to be malignant.",
    "-0.20 < smoothness_se <= 0.35": "Smaller values of 'smoothness_se' within this range have a negative contribution, indicating that tumors with less variability in smoothness are less likely to be malignant.",
    "-0.67 < concave points_se <= -0.13": "Higher values of 'concave points_se' within this range have a positive contribution, indicating that tumors with more concave points in their contours are more likely to be malignant.",
    "-0.24 < concave points_worst <= 0.72": "Higher values of 'concave points_worst' within this range have a positive contribution, indicating that tumors with more concave points in their worst areas are more likely to be malignant.",
    "-0.23 < concavity_worst <= 0.54": "Smaller values of 'concavity_worst' within this range contribute negatively, suggesting that tumors with less severe concave portions in their worst areas are less likely to be malignant.",
    "-0.66 < symmetry_se <= -0.23": "Smaller values of 'symmetry_se' within this range have a negative contribution, indicating that tumors with less variation in symmetry are less likely to be malignant."
}

# Print out the detailed explanations
print("\nDetailed Feature Explanations:")
for feature, contribution in feature_contributions:
    explanation = feature_explanations.get(feature, "No specific explanation available for this feature.")
    print(f"{feature} (Contribution: {contribution:.4f}): {explanation}")


# In[18]:


# Highlighting the top 5 features
print("\nTop 5 Most Important Features for the specific instance:")
top_5_features = sorted(feature_contributions, key=lambda x: np.abs(x[1]), reverse=True)[:5]
for feature, contribution in top_5_features:
    print(f"{feature}: {contribution:.4f}")


# In[ ]:





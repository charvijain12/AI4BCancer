#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


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


# In[10]:


# Implement Support Vector Machine (SVM)
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, y_train)


# In[11]:


# Evaluate the model on the test set
y_pred = svm_model.predict(X_test_scaled)


# In[12]:


# Evaluate the model on the test set
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)


# In[14]:


# Create a Kernel SHAP explainer
explainer = shap.KernelExplainer(svm_model.decision_function, X_train_scaled)

# Calculate SHAP values - Note: This can be computationally expensive
shap_values = explainer.shap_values(X_test_scaled)


# In[15]:


# Visualize the SHAP values - here we visualize the first prediction
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0], X_test_scaled[0], feature_names=X.columns)


# In[16]:


# Textual Explanation
# Calculate the mean absolute SHAP values for each feature
shap_values_mean = np.abs(shap_values).mean(axis=0)

# Create a DataFrame for easier handling
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'mean_shap_value': shap_values_mean
}).sort_values(by='mean_shap_value', ascending=False)


# In[18]:


# Print the sorted feature importances along with their mean SHAP values
print("Sorted Feature Importances from SHAP Analysis:")
for index, row in feature_importance_df.iterrows():
    print(f"{row['feature']} (Mean SHAP Value: {row['mean_shap_value']:.4f})")


# In[19]:


# Detailed Feature Explanations based on SHAP values
feature_explanations = {
    "concave points_mean": "Average number of concave portions of the contour of the tumor. Indicates complexity and irregularity of tumor shape.",
    "texture_worst": "Standard deviation of gray-scale values in the worst area. Reflects variance in cell structure and density.",
    "radius_se": "Standard error for the mean of distances from the center to points on the perimeter. Indicates variability in tumor size.",
    "concavity_worst": "Severity of concave portions of the contour in the worst segment. Suggests tumor irregularities and potential malignancy.",
    "compactness_mean": "Mean compactness of the tumor, defined as (perimeter^2 / area - 1.0). Reflects the complexity and density of the tumor.",
    "symmetry_worst": "Symmetry of the tumor in its worst segment. Asymmetry can be an indicator of malignancy.",
    "concavity_mean": "Average severity of concave portions of the tumor contour. Indicates depth of indentations in the tumor.",
    "radius_worst": "Mean of distances from center to points on the perimeter in the tumor’s worst area. Reflects overall tumor size.",
    "area_se": "Standard error of the area of the tumor. Shows variability in tumor size.",
    "area_worst": "The area of the tumor's worst segment. Larger areas can be indicative of advanced stages.",
    "fractal_dimension_worst": "‘Coastline approximation’ - 1, measured in the tumor's worst area. Indicates complexity of tumor border.",
    "fractal_dimension_se": "Standard error of the 'coastline approximation'. Shows variability in the complexity of tumor contour.",
    "texture_se": "Standard error of gray-scale values. Indicates variability in texture and cell structures.",
    "symmetry_se": "Standard error of tumor symmetry. Variability in symmetry can suggest differences in tumor growth.",
    "radius_mean": "Average of distances from the center to points on the perimeter. Indicates general tumor size.",
    "smoothness_se": "Standard error in local variation of radius lengths. Shows variability in tumor texture.",
    "compactness_se": "Standard error of the compactness. Indicates variability in the density and complexity of the tumor.",
    "perimeter_mean": "Average size of the perimeter of the tumor. Reflects general dimensions of the tumor.",
    "smoothness_worst": "Local variation in radius lengths in the worst segment. Indicates changes in texture in the most severe part.",
    "area_mean": "Average area of the tumor. Provides an overall measure of tumor size.",
    "perimeter_se": "Standard error of the size of the tumor perimeter. Reflects variability in tumor shape.",
    "concave points_se": "Standard error of the number of concave portions of the contour. Indicates variability in tumor contour irregularities.",
    "concavity_se": "Standard error of the severity of concave portions of the contour. Reflects variability in indentations.",
    "symmetry_mean": "Average symmetry of the tumor. Asymmetry can be a sign of malignancy.",
    "compactness_worst": "Compactness of the tumor in its worst area. Indicates density and complexity in the most severe part.",
    "concave points_worst": "Number of concave portions in the worst segment. Reflects irregularities in the most severe part of the tumor.",
    "perimeter_worst": "Perimeter size in the tumor’s worst segment. Indicates size and extent in the most severe part.",
    "fractal_dimension_mean": "‘Coastline approximation’ - 1, average measure. Reflects overall complexity of tumor contour.",
    "texture_mean": "Average standard deviation of gray-scale values. Reflects variation in cell structures.",
    "smoothness_mean": "Mean local variation in radius lengths. Indicates overall tumor texture."
}

# Print out the detailed explanations
print("\nDetailed Feature Explanations:")
for feature_name, shap_value in feature_importance_df.itertuples(index=False):
    explanation = feature_explanations.get(feature_name, "No explanation available")
    print(f"{feature_name} (Mean SHAP Value: {shap_value:.4f}): {explanation}")


# In[21]:


# Extracting the top 5 features from the feature importance DataFrame
top_5_features = feature_importance_df.head(5)

print("\nTop 5 Most Important Features for Predicting Breast Cancer:")
for index, row in top_5_features.iterrows():
    print(f"- {row['feature']} (Mean SHAP Value: {row['mean_shap_value']:.4f})")


# In[ ]:





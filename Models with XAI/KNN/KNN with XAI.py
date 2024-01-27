#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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


# In[9]:


# Train a KNN model
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_scaled, y_train)


# In[10]:


# Evaluate the model on the test set
y_pred = knn_model.predict(X_test_scaled)


# In[11]:


# Print accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)


# In[12]:


import lime
import lime.lime_tabular

# Create a LimeTabularExplainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=X.columns,
    class_names=['Benign', 'Malignant'],
    mode='classification'
)

# Choose a specific instance to explain
instance_index = 10  # for example
exp = explainer.explain_instance(X_test_scaled[instance_index], knn_model.predict_proba, num_features=len(X.columns))

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


# In[15]:


# Detailed explanations for each feature based on the LIME output
feature_explanations = {
    "texture_worst > 0.69": "Higher values of worst texture indicate more variation in the sizes and shapes of the cells in the worst-affected area, potentially signaling a more aggressive form of cancer.",
    "texture_mean > 0.56": "Increased mean texture suggests greater irregularity in the cell structures, often associated with malignant tumors.",
    "concave points_worst <= -0.77": "Fewer concave points in the worst segment of the tumor could imply a smoother contour, typically seen in benign tumors.",
    "concavity_mean <= -0.75": "Lower average concavity may indicate shallower and less frequent indentations in the contour of the tumor, often associated with benign cases.",
    "symmetry_worst <= -0.65": "Lower symmetry in the worst part of the tumor might suggest less aggressive tumor characteristics.",
    "radius_se <= -0.59": "Smaller standard error in radius indicates more uniformity in the tumor size, which can be a characteristic of benign tumors.",
    "concave points_mean <= -0.73": "Fewer average concave points might indicate a smoother, less irregular tumor, often associated with benign cases.",
    "perimeter_worst <= -0.69": "A smaller perimeter in the worst segment of the tumor could be indicative of a less advanced stage of cancer.",
    "radius_worst <= -0.66": "Smaller worst radius suggests a smaller tumor size in its most severe state, potentially indicating a less advanced cancer stage.",
    "area_worst <= -0.64": "A smaller area in the worst segment of the tumor might indicate a less aggressive form of cancer.",
    "area_se <= -0.46": "A smaller standard error of the tumor area may indicate a more uniform tumor size, commonly associated with benign characteristics.",
    "perimeter_se <= -0.58": "A smaller standard error in the tumor's perimeter suggests uniformity in shape, which might be indicative of non-malignant growth.",
    "fractal_dimension_mean <= -0.71": "Lower average fractal dimension implies less complexity in the tumor's texture, often found in benign cases.",
    "symmetry_se <= -0.66": "Lower standard error in symmetry could indicate consistent symmetry across the tumor, typically seen in benign tumors.",
    "fractal_dimension_se <= -0.57": "A lower standard error in fractal dimension suggests less variability in the tumor's complexity, potentially pointing to benign nature.",
    "concavity_worst <= -0.76": "Reduced severity of concave portions in the tumor's worst area may suggest a less aggressive tumor.",
    "compactness_se <= -0.69": "Lower standard error in compactness indicates uniformity in the tumor's density, often related to benign tumors.",
    "radius_mean <= -0.68": "A smaller average radius of the tumor can be indicative of a smaller, potentially benign tumor.",
    "perimeter_mean <= -0.69": "Smaller average perimeter size might suggest a smaller, less aggressive tumor.",
    "fractal_dimension_worst <= -0.72": "Reduced fractal dimension in the tumor's worst area could indicate less complexity, often seen in benign tumors.",
    "smoothness_worst <= -0.74": "Lower smoothness in the tumor's most severe part might indicate a more uniform texture, typically associated with benign tumors.",
    "area_mean <= -0.66": "A smaller average tumor area can suggest a less advanced stage of cancer, possibly benign.",
    "concave points_se <= -0.67": "Fewer concave points might indicate a smoother contour of the tumor, commonly seen in benign cases.",
    "smoothness_se <= -0.63": "Lower standard error in smoothness indicates a more consistent texture across the tumor, often associated with benign characteristics.",
    "texture_se <= -0.20": "Lower texture variation can suggest less variability in cell structures, potentially benign.",
    "smoothness_mean <= -0.71": "Lower average smoothness of the tumor might indicate a more uniform texture, which is often found in benign tumors.",
    "compactness_mean <= -0.78": "Reduced compactness could imply a less dense and less complex tumor, typically associated with benign characteristics.",
    "compactness_worst <= -0.70": "Lower compactness in the tumor's worst area might suggest a less dense and less complex tumor, often seen in benign cases.",
    "concavity_se <= -0.55": "Lower variability in the severity of concave portions could indicate a smoother, more uniform tumor, potentially benign.",
    "symmetry_mean <= -0.70": "Lower average symmetry might suggest less aggressive tumor characteristics, often associated with benign tumors."
}

# Print out the detailed explanations
print("\nDetailed Feature Explanations:")
for feature, contribution in feature_contributions:
    explanation = feature_explanations.get(feature, "No specific explanation available for this feature.")
    print(f"{feature} (Contribution: {contribution:.4f}): {explanation}")


# In[16]:


# Highlighting the top 5 features
print("\nTop 5 Most Important Features for the specific instance:")
top_5_features = sorted(feature_contributions, key=lambda x: np.abs(x[1]), reverse=True)[:5]
for feature, contribution in top_5_features:
    print(f"{feature}: {contribution}")


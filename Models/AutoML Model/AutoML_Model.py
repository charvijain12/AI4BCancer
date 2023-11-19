#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tpot import TPOTClassifier
from sklearn.metrics import classification_report, accuracy_score


# In[2]:


# Load the Wisconsin Breast Cancer dataset
file_path = r'C:\Users\Shaurya\Desktop\BCancerAI\Dataset\data.csv'
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


# Instantiate TPOTClassifier
tpot_classifier = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42, config_dict='TPOT sparse')


# In[9]:


# Fit the model
tpot_classifier.fit(X_train, y_train)


# In[10]:


# Make predictions
y_pred = tpot_classifier.predict(X_test)


# In[11]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)


# In[12]:


# Export the pipeline
tpot_classifier.export('tpot_breast_cancer_pipeline.py')


# In[ ]:





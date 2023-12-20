#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score


# In[2]:


# Load the dataset
df = pd.read_csv('C:\\Users\\Shaurya\\Downloads\\data.csv')
df = df.drop(['id', 'Unnamed: 32'], axis=1)
X = df.drop('diagnosis', axis=1)
y = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)


# In[3]:


# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[5]:


# Initialize the online learning model with the correct loss parameter
model = SGDClassifier(loss='log_loss')  # Corrected loss parameter for logistic regression

# Simulate online learning
# Here we'll update the model in batches
batch_size = 10
for i in range(0, X_train.shape[0], batch_size):
    X_batch = X_train[i:i + batch_size]
    y_batch = y_train[i:i + batch_size]
    model.partial_fit(X_batch, y_batch, classes=np.unique(y))


# In[6]:


# Evaluate the model
X_test_scaled = scaler.transform(X_test)  # Ensure test data is scaled
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)


# In[ ]:





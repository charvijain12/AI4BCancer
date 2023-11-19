#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# In[2]:


# Load the Wisconsin Breast Cancer dataset
file_path = r'C:\Users\Shaurya\Desktop\BCancerAI\Dataset\data.csv'
df = pd.read_csv(file_path)


# In[3]:


# Drop unnecessary columns
df = df.drop(columns=['id', 'Unnamed: 32'])


# In[4]:


# Encode the 'diagnosis' column (M/B) to numerical values (1/0)
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


# Display the count of null values in each column after preprocessing
print("\nNull values after preprocessing:")
print(df.isnull().sum())


# In[10]:


# Summary statistics for cleaned and preprocessed data
print("\nSummary statistics for preprocessed data:")
print(df.describe())


# In[11]:


# Information about the preprocessed dataset
print("\nInformation about the preprocessed dataset:")
print(df.info())


# In[12]:


# Visualize the distribution of the diagnosis column in the preprocessed dataset
sns.countplot(x='diagnosis', data=df)
plt.title('Distribution of Diagnosis (Preprocessed Data)')
plt.show()


# In[13]:


# Correlation matrix heatmap for preprocessed data
correlation_matrix_preprocessed = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_preprocessed, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix (Preprocessed Data)')
plt.show()


# In[14]:


# Pairplot (scatterplot matrix) for preprocessed data
sns.pairplot(df, hue='diagnosis')
plt.suptitle('Pairplot of Variables (Preprocessed Data)')
plt.show()


# In[ ]:





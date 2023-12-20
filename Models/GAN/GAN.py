#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv('C:\\Users\\Shaurya\\Downloads\\data.csv')
df = df.drop(['id', 'Unnamed: 32'], axis=1)
X = df.drop('diagnosis', axis=1)
y = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the GAN and its components
def build_generator(latent_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(X_train.shape[1], activation='tanh'))
    return model

def build_discriminator():
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1]))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

latent_dim = 100
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
generator = build_generator(latent_dim)
discriminator.trainable = False
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

# Function to train GAN
def train_gan(generator, discriminator, gan, X, epochs, batch_size, latent_dim):
    half_batch = batch_size // 2
    for epoch in range(epochs):
        # Train the discriminator
        idx = np.random.randint(0, X.shape[0], half_batch)
        real_samples = X[idx]
        real_y = np.ones((half_batch, 1))
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_samples = generator.predict(noise)
        fake_y = np.zeros((half_batch, 1))
        d_loss_real = discriminator.train_on_batch(real_samples, real_y)
        d_loss_fake = discriminator.train_on_batch(fake_samples, fake_y)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_y = np.ones((batch_size, 1))  # Generator wants discriminator to think samples are real
        g_loss = gan.train_on_batch(noise, g_y)

        # Print progress
        print(f"Epoch {epoch+1}/{epochs} \t Discriminator Loss: {d_loss[0]} \t Generator Loss: {g_loss}")

# Train GAN
train_gan(generator, discriminator, gan, X_train, epochs=10000, batch_size=32, latent_dim=latent_dim)


# In[7]:


# Use the discriminator for predictions and generate classification report
X_test_scaled = scaler.transform(X_test)  # Ensure test data is scaled
y_pred_prob = discriminator.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()  # Convert probabilities to class labels
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)


# In[ ]:





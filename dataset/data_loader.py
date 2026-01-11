from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# Load the saved NumPy arrays
# A = np.load("dataset/A.npy")
letter = np.load("dataset/V.npy")

# print(A.shape)
print(letter.shape)

# Combine the two arrays
# allData = np.vstack((A, notA))

# Split features and labels for training
# X = allData[:, :-1]  # Features
# Y = allData[:, -1]   # Labels
# 
# scaler = StandardScaler() 
# X_scaled = scaler.fit_transform(X)  # Feature scaling
# __all__ = ['X_scaled', 'Y', 'scaler']
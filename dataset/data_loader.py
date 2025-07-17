from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the saved NumPy arrays
A = np.load("A.npy")
notA = np.load("notA.npy")

print(A.shape)
print(notA.shape)

# Combine the two arrays
allData = np.vstack((A, notA))

# Split features and labels for training
X = allData[:, :-1]  # Features
Y = allData[:, -1]   # Labels

X_scaled = StandardScaler().fit_transform(X)  # Feature scaling
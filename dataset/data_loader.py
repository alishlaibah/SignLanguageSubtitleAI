from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the saved NumPy arrays
A = np.load("A.npy")
notA = np.load("notA.npy")

# print("A shape:", A.shape)
# print("notA shape:", notA.shape)

allData = np.vstack((A, notA))
X = allData[:, :-1]  # Features
Y = allData[:, -1]   # Labels

print(X)
print(Y)

scaler = StandardScaler()
X = scaler.fit_transform(X)  # Standardize coordinates
print(X)
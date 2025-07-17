from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the saved NumPy arrays
A = np.load("A.npy")
notA = np.load("notA.npy")

# print("A shape:", A.shape)
# print("notA shape:", notA.shape)

# print(A)
# print(notA)

allData = np.vstack((A, notA))
X = allData[:, :-1]  # Features
Y = allData[:, -1]   # Labels

print(X)
print(Y)


X_scaled = StandardScaler().fit_transform(X)  # Features
print(X_scaled)
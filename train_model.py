import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = []
y = []

data_path = "data"

for file in os.listdir(data_path):
    if file.endswith(".npy"):
        label = file.split(".")[0]
        data = np.load(os.path.join(data_path, file))

        for sample in data:
            X.append(sample)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Total samples:", X.shape)

# Split for validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=300)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Validation Accuracy:", accuracy)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")

"""python train_model.py"""
"""python collect_data.py"""
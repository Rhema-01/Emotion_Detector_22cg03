import os
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# create fake data (e.g. 3 classes)
n_samples = 150
img_size = 48*48
X = np.random.randn(n_samples, img_size).astype(float)
y = np.array(["happy"]*(n_samples//3) + ["sad"]*(n_samples//3) + ["neutral"]*(n_samples - 2*(n_samples//3)))

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=50, random_state=42)
mlp.fit(Xs, y)

joblib.dump(mlp, os.path.join(MODEL_DIR, "mlp_emotion.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print("Dummy model + scaler saved to", MODEL_DIR)
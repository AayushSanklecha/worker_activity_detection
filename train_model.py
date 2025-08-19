import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold
import joblib
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load dataset
data = np.load("dataset/stanford40_idle_active.npz")
X, y = data["X"], data["y"]

# Preprocess
X = preprocess_input(X.astype("float32"))

# Feature extractor (pretrained MobileNetV2, no top layer)
base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(128,128,3))
features = base_model.predict(X, batch_size=32, verbose=1)

# K-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
accuracies = []

for train_index, test_index in skf.split(features, y):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    
    print(f"Fold {fold} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    fold += 1

print(f"\nAverage Accuracy across folds: {np.mean(accuracies):.4f}")

# Final training on all data
clf_final = LogisticRegression(max_iter=1000)
clf_final.fit(features, y)

# Save model
joblib.dump(clf_final, "models/stanford40_activity_clf.pkl")
np.save("models/feature_mean.npy", features.mean(axis=0))  
print("Final Model saved.")

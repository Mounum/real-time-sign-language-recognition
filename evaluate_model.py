import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ===== LOAD MODEL =====
model = tf.keras.models.load_model("sign_language_model.h5")

# ===== DATASET PATH =====
data_dir = "dataset"

# ===== IMAGE SIZE =====
img_size = 224

# ===== CLASS NAMES =====
class_names = sorted(os.listdir(data_dir))

X = []
y = []

# ===== LOAD IMAGES =====
for label, folder in enumerate(class_names):
    folder_path = os.path.join(data_dir, folder)
    
    for img_name in os.listdir(folder_path)[:50]:  # limit for speed
        img_path = os.path.join(folder_path, img_name)
        
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0
        
        X.append(img)
        y.append(label)

X = np.array(X)
y = np.array(y)

# ===== PREDICTIONS =====
pred = model.predict(X)
pred_classes = np.argmax(pred, axis=1)

# ===== CONFUSION MATRIX =====
cm = confusion_matrix(y, pred_classes)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")
plt.show()

print("✅ Confusion matrix generated")

# ===== CLASS ACCURACY GRAPH =====
class_accuracy = cm.diagonal() / cm.sum(axis=1)

plt.figure(figsize=(12,6))
plt.bar(class_names, class_accuracy)
plt.xticks(rotation=45)
plt.title("Accuracy per Sign Class")
plt.xlabel("Sign Class")
plt.ylabel("Accuracy")

plt.savefig("class_accuracy.png")
plt.show()

print("✅ Class accuracy graph generated")
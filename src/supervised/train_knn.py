from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load data
x = pd.read_csv("../../data/supervised/classification_training_data/V04 Fan/feature_VBL-VA001.csv", header=None)
y = pd.read_csv("../../data/supervised/classification_training_data/V04 Fan/label_VBL-VA001.csv", header=None)
y = pd.Series.ravel(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, shuffle=True
)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)

# Print accuracies
print("KNN Accuracy on Train Data: {:.2f}".format(knn.score(X_train, y_train)))
print("KNN Accuracy on Test Data: {:.2f}".format(knn.score(X_test, y_test)))

# Save the model
joblib.dump(knn, '../../models/supervised/knn_model.joblib')

# Plot confusion matrix
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred, normalize='true')
class_names = ['Normal', 'Misalignment', 'Unbalance', 'Looseness', 'Impact']

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap='YlGn', values_format='.2f', xticks_rotation=45)
ax.set_title('KNN Confusion Matrix')

plt.tight_layout()
plt.savefig('../../plots/supervised/knn_cm.png')
plt.show()
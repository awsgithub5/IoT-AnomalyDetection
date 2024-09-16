from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load data
x = pd.read_csv("../../data/supervised/classification_training_data/Fan/feature_VBL-VA001.csv", header=None)
y = pd.read_csv("../../data/supervised/classification_training_data/Fan/label_VBL-VA001.csv", header=None)
y = pd.Series.ravel(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

# Perform Grid Search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get best model
best_rf = grid_search.best_estimator_

# Print best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Print accuracies
print("RF Accuracy on Train Data: {:.2f}".format(best_rf.score(X_train, y_train)))
print("RF Accuracy on Test Data: {:.2f}".format(best_rf.score(X_test, y_test)))

# Print classification report
y_pred = best_rf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(best_rf, '../../models/supervised/rf_model.joblib')

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred, normalize='true')
class_names = ['Normal', 'Misalignment', 'Unbalance', 'Looseness', 'Impact']

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap='YlGn', values_format='.2f', xticks_rotation=45)
ax.set_title('Random Forest Confusion Matrix')

plt.tight_layout()
plt.savefig('../../plots/supervised/rf_cm.png')
plt.show()

# Plot feature importances
feature_importance = best_rf.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(pos, feature_importance[sorted_idx], align='center')
ax.set_yticks(pos)
ax.set_yticklabels(sorted_idx)
ax.set_xlabel('Feature Importance')
ax.set_title('Random Forest Feature Importances')

plt.tight_layout()
plt.savefig('../../plots/supervised/rf_feature_importances.png')
plt.show()



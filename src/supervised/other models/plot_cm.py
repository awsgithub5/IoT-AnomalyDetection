# script to plot best confusion matrx
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# load data
# load data hasil ekstraksi fitur fft
x = pd.read_csv("data/classification_training_data/V04 Fan/feature_VBL-VA001.csv", header=None)

# load label
y = pd.read_csv("data/classification_training_data/V04 Fan/label_VBL-VA001.csv", header=None)

# make 1D array to avoid warning
y = pd.Series.ravel(y)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, shuffle=True
)

# KNN
knn = KNeighborsClassifier(n_neighbors=2)
out_knn = knn.fit(X_train, y_train)
print("KNN Accuracy on Train Data: {}".format(knn.score(X_train, y_train)))
print("KNN Accuracy on Test Data: {}".format(knn.score(X_test, y_test)))

# SVM Machine Learning
svm = SVC(C=86, kernel='rbf', class_weight='balanced', random_state=None)
out_svm = svm.fit(X_train, y_train)
print("SVM accuracy is {} on Train Dataset".format(svm.score(X_train, y_train)))
print("SVM accuracy is {} on Test Dataset".format(svm.score(X_test, y_test)))

# Naive Bayes
model = GaussianNB(var_smoothing=1e-11)
out_gnb = model.fit(X_train, y_train)
gnb_pred = model.predict(X_test)

print("NB accuracy is {} on Train Dataset".format(model.score(X_train,
                                                              y_train)))
print("NB accuracy is {} on Test Dataset".format(model.score(X_test, y_test)))


# class for all
class_names = ['Normal', 'Misalignment', 'Unbalance', 'Looseness','Impact']

# plt.figure()
# plt.subplot(1, 3, 1)
# plot_confusion_matrix(out_knn, X_test, y_test, display_labels=class_names,
#                       xticks_rotation=45, cmap=plt.cm.Greens,
#                       values_format='.2f',
#                       normalize='true')
# plt.subplot(1, 3, 2)
# plot_confusion_matrix(out_svm, X_test, y_test, display_labels=class_names,
#                       xticks_rotation=45, cmap=plt.cm.Greens,
#                       values_format='.2f',
#                       normalize='true')
# plt.subplot(1, 3, 3)
# plot_confusion_matrix(out_gnb, X_test, y_test, display_labels=class_names,
#                       xticks_rotation=45, cmap=plt.cm.Greens,
#                       values_format='.2f',
#                       normalize='true')

# plt.savefig('cm.svg')

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10))

for out, ax, title in zip([out_knn, out_svm, out_gnb], axes.flatten(), ['KNN', 'SVM', 'Naive Bayes']):
    y_pred = out.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='YlGn', values_format='.2f', xticks_rotation=45, colorbar=False)
    ax.set_title(f'{title} Confusion Matrix')

plt.tight_layout()
plt.savefig('cm.svg')
plt.show()

# plt.savefig('cm.svg')
plt.show()

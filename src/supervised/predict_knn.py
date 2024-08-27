import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def load_and_predict(model_path, data_folder):
    # Load the model
    knn = joblib.load(model_path)
    
    # Create necessary folders
    os.makedirs('../../plots/supervised/classes', exist_ok=True)
    
    class_names = ['Normal', 'Misalignment', 'Unbalance', 'Looseness', 'Impact']
    
    # Process each file in the data folder
    for filename in os.listdir(data_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_folder, filename)
            
            # Load and predict
            X = pd.read_csv(file_path, header=None)
            X = X.iloc[:, 4:] 
            y_pred = knn.predict(X)
            
            # Segregate into classes
            for i, class_name in enumerate(class_names):
                class_data = X[y_pred == i]
                os.makedirs(f'../../plots/supervised/classes/{class_name}', exist_ok=True)
                class_data.to_csv(f'../../plots/supervised/classes/{class_name}/{filename}', index=False, header=False)
            
            # Plot and save
            plt.figure(figsize=(10, 6))
            plt.plot(X.iloc[:, 0], X.iloc[:, 1], 'b.')
            plt.title(f'Scatter plot for {filename}')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.savefig(f'../../plots/supervised/{filename[:-4]}_scatter.png')
            plt.close()

            # If true labels are available, create confusion matrix
            if os.path.exists(os.path.join(data_folder, f'label_{filename}')):
                y_true = pd.read_csv(os.path.join(data_folder, f'label_{filename}'), header=None)
                y_true = pd.Series.ravel(y_true)
                cm = confusion_matrix(y_true, y_pred, normalize='true')
                
                fig, ax = plt.subplots(figsize=(10, 8))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
                disp.plot(ax=ax, cmap='YlGn', values_format='.2f', xticks_rotation=45)
                ax.set_title(f'Confusion Matrix for {filename}')
                plt.tight_layout()
                plt.savefig(f'plots/{filename[:-4]}_cm.png')
                plt.close()

if __name__ == "__main__":
    model_path = '../../models/supervised/knn_model.joblib'
    data_folder = '../../data/unsupervised/output_abnormal'
    load_and_predict(model_path, data_folder)
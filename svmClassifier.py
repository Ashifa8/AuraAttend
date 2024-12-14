import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib  # For saving the trained model

# Load the embeddings from the CSV file
csv_file = 'face_embeddings.csv'
data = pd.read_csv(csv_file)

# Separate the labels (person names) and embeddings (numerical data)
labels = data['Person'].values
embeddings = data.drop('Person', axis=1).values

# Convert the string labels to numeric labels (for classification)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.2, random_state=42)

# Initialize the SVM classifier
svm_clf = svm.SVC(kernel='linear')

# Train the classifier
print("Training the SVM classifier...")
svm_clf.fit(X_train, y_train)

# Make predictions
y_pred = svm_clf.predict(X_test)

# Print classification report
print("Classification report:")
print(classification_report(y_test, y_pred, labels=np.unique(y), target_names=label_encoder.classes_))

# Save the trained model to a file using joblib
joblib.dump(svm_clf, 'svm_face_recognition_model.pkl')
print("Model saved to 'svm_face_recognition_model.pkl'")

# Optional: If you want to load the model later for predictions, you can use:
# svm_clf = joblib.load('svm_face_recognition_model.pkl')

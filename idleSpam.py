import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score
from sklearn.svm import SVC

# Load mail data
raw_mail_data = pd.read_csv('C:/Users/merlin cj/Downloads/mail_data (1).csv')
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')

# Label Encoding
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

X = mail_data['Message']
Y = mail_data['Category']

# Train-test split
X_Train, X_test, Y_Train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Feature Extraction
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_feature = feature_extraction.fit_transform(X_Train)
X_test_feature = feature_extraction.transform(X_test)

# Convert Y_train and Y_test to integers
Y_Train = Y_Train.astype('int')
Y_test = Y_test.astype('int')

# Train Support Vector Machine
svc = SVC(kernel='sigmoid', gamma=1.0)
svc.fit(X_train_feature, Y_Train)
y_pred1 = svc.predict(X_test_feature)
accuracy = accuracy_score(Y_test, y_pred1)
precision = precision_score(Y_test, y_pred1)

# Function to classify mail
def classify_mail():
    input_mail = [mail_entry.get()]
    input_data_feature = feature_extraction.transform(input_mail)
    prediction = svc.predict(input_data_feature)

    if prediction == [1]:
        result_label.config(text="This is a Ham Mail.")
    else:
        result_label.config(text="This is a Spam Mail.")

# GUI
root = tk.Tk()
root.title("Mail Classifier")

mail_label = ttk.Label(root, text="Enter Mail Content:")
mail_label.grid(row=0, column=0, padx=10, pady=10)

mail_entry = ttk.Entry(root, width=50)
mail_entry.grid(row=0, column=1, padx=10, pady=10)

classify_button = ttk.Button(root, text="Classify", command=classify_mail)
classify_button.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

result_label = ttk.Label(root, text="")
result_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

accuracy_label = ttk.Label(root, text=f"Accuracy: {accuracy}")
accuracy_label.grid(row=3, column=0, columnspan=2, padx=10, pady=5)

precision_label = ttk.Label(root, text=f"Precision: {precision}")
precision_label.grid(row=4, column=0, columnspan=2, padx=10, pady=5)

root.mainloop()

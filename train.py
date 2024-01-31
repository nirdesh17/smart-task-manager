import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

#Load Dataset
data = pd.read_csv('Data.csv')
# print(data.head())

#Data Preprocessing
data=data.dropna() #drop null values
data = data.apply(lambda x: x.astype(str).str.lower())
# print(data.head())

#Encoding priority values
priority_mapping={'low':0,'medium':1,'high':2}
data['priority'] = data['priority'].map(priority_mapping)
# print(data.head())

X=data['description']
y=data['priority']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

# Save the model and vectorizer as pickle files
joblib.dump(model, 'trained_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')



y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy}')
# print('Classification Report:')
# print(classification_report(y_test, y_pred))

# def predict_priority(task_description):
#     task_description_tfidf = tfidf_vectorizer.transform([task_description])
#     predicted_priority = model.predict(task_description_tfidf)
#     return predicted_priority[0]

# Example usage:
# Example usage:
# new_task_description = "call client"
# predicted_priority = predict_priority(new_task_description)

# priority_mapping_reverse = {v: k for k, v in priority_mapping.items()}
# predicted_priority_label = priority_mapping_reverse[predicted_priority]

# print(f'Predicted Priority: {predicted_priority_label}')


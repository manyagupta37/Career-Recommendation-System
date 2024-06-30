# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 18:51:46 2024

@author: hp
"""

import pandas as pd
import os
data = pd.read_csv("C:\\Users\hp\Desktop\career_recommender.csv",encoding='latin1')
data1=data.copy()
data1.head()

#%%
data1.drop(columns=['name','certification courses','working'],axis=1, inplace=True)


#%%
data1['career aspiration'].value_counts()

#%%
len(data1['career aspiration'].unique())

#%%
from sklearn import preprocessing 

encoders={}


label_encoder = preprocessing.LabelEncoder() 

columns_to_encode = ['UG course', 'UG specialization', 'interests', 'skills', 'certificate course title.']

for column in columns_to_encode:
    label_encoder = preprocessing.LabelEncoder() 
    data1[column] = label_encoder.fit_transform(data1[column])
    encoders[column] = label_encoder

data1.head()

#%%
data1 = data1.dropna(subset=['career aspiration'])

# Display the first few rows to confirm
data1.head()

#%%
gender_map = {'Male':0, 'Female':1}
data1['gender']=data1['gender'].map(gender_map)
data1.head()

#%%
target_encoder = preprocessing.LabelEncoder()
data1['career aspiration'] = target_encoder.fit_transform(data1['career aspiration'])
encoders['career aspiration'] = target_encoder

#%%
data1.shape

#%%
X = data1.drop('career aspiration',axis=1)
y = data1['career aspiration']

#%%
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

#%%
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_imputed,y)

#%%
y_resampled.value_counts()

#%%
y_resampled.shape


#%%
from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X_resampled,y_resampled,test_size=0.2, random_state=42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape 

#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

models = {
    "Logistic Regression": LogisticRegression(),
    "Support Vector Classifier": SVC(),
    "Random Forest Classifier": RandomForestClassifier(),
    "K Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "Gaussian Naive Bayes": GaussianNB(),
    "AdaBoost Classifier": AdaBoostClassifier(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "XGBoost Classifier": XGBClassifier(use_label_encoder=False,eval_metric='mlogloss')
    }

for name,model in models.items():
    
    print("Model:",name)
    model.fit(X_train_scaled,y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test,y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print("Accuracy:",accuracy)
    print("Classification Report:\n",classification_rep)
    print("Confusion Matrix:\n",conf_matrix)
    
#%%
model= RandomForestClassifier()

model.fit(X_train_scaled,y_train)
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test,y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:",accuracy)
print("Classification Report:\n",classification_rep)
print("Confusion Matrix:\n",conf_matrix)

#%%
X_test_scaled[10].reshape(1,-1)

#%%
print("Predicted Label:",model.predict(X_test_scaled[0].reshape(1,-1)))
print("Actual label:", y_test.iloc[0])


#%%
for idx in [0, 10, 50, 100, 200]:
    print(f"Index {idx} - Predicted Label: {model.predict(X_test_scaled[idx].reshape(1, -1))[0]}, Actual Label: {y_test.iloc[idx]}")



#%%
import os
import pickle

os.makedirs("Models",exist_ok=True)

pickle.dump(encoders, open("Models/encoders.pkl", 'wb'))
pickle.dump(scaler,open("Models/scaler.pkl",'wb'))
pickle.dump(model,open("Models/model.pkl",'wb'))

encoders = pickle.load(open("Models/encoders.pkl", 'rb'))
scaler = pickle.load(open("Models/scaler.pkl",'rb'))
model = pickle.load(open("Models/model.pkl",'rb'))

#%%
import numpy as np

# Ensure the class names are correct
print(encoders['career aspiration'].classes_)

# Update this list based on the output from the above print statement
class_names = encoders['career aspiration'].classes_

class_names = ['Web Developer' if x == 'Student (Unemployed)' else x for x in class_names]

print(class_names)
def Recommendations(gender,UG_course,UG_specialization, interests,skills, UG_Percentage,certificate_course_title):
    
    
    gender_map = {'Male': 0, 'Female': 1}
    gender_encoded = gender_map[gender]
    UG_course_encoded = encoders['UG course'].transform([UG_course])[0]
    UG_specialization_encoded = encoders['UG specialization'].transform([UG_specialization])[0]
    interests_encoded = encoders['interests'].transform([interests])[0]
    skills_encoded = encoders['skills'].transform([skills])[0]
    certificate_course_title_encoded = encoders['certificate course title.'].transform([certificate_course_title])[0]
    
    feature_array = np.array([[gender_encoded, UG_course_encoded, UG_specialization_encoded, interests_encoded, skills_encoded, UG_Percentage, certificate_course_title_encoded]])
    
    scaled_features = scaler.transform(feature_array)
    
    
    probabilities = model.predict_proba(scaled_features)
    
    top_classes_idx = np.argsort(-probabilities[0])[:3]
    
    top_classes_names_probs = [(class_names[idx], probabilities[0][idx]) for idx in top_classes_idx]
    
    return top_classes_names_probs
    
    
#%%

final_recommendations = Recommendations(gender='Female',
                                        UG_course='BE',
                                        UG_specialization='Computer Science Engineering',
                                        interests='Technology',
                                        skills='Python',
                                        UG_Percentage=72,
                                        certificate_course_title='Data Science')
print("Top recommended carrer with prababilities:")
for class_name, probability in final_recommendations:
    print(f"{class_name} with probability {probability}")
    
#%%
final_recommendations = Recommendations(gender='Male',
                                        UG_course='B.Tech',
                                        UG_specialization='Mechanical Engineering',
                                        interests='Technology',
                                        skills='Programming Language skills',
                                        UG_Percentage=71,
                                        certificate_course_title='Python')
print("Top recommended carrer with prababilities:")
for class_name, probability in final_recommendations:
    print(f"{class_name} with probability {probability}")
    
#%%
import sklearn
print(sklearn.__version__)

#%%
import numpy as np
print(np.__version__)

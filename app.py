import time

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

import numpy as np

print(np.__version__)

app = Flask(__name__)

scaler = pickle.load(open("Models/scaler.pkl", 'rb'))
model = pickle.load(open("Models/model.pkl", 'rb'))
encoders = pickle.load(open("Models/encoders.pkl", 'rb'))

class_names = encoders['career aspiration'].classes_

class_names = ['Web Developer' if x == 'Student (Unemployed)' else x for x in class_names]


def Recommendations(gender, UG_course, UG_specialization, interests, skills, UG_Percentage, certificate_course_title):
    gender_map = {'Male': 0, 'Female': 1}
    gender_encoded = gender_map[gender]
    UG_course_encoded = encoders['UG course'].transform([UG_course])[0]
    UG_specialization_encoded = encoders['UG specialization'].transform([UG_specialization])[0]
    interests_encoded = encoders['interests'].transform([interests])[0]
    skills_encoded = encoders['skills'].transform([skills])[0]
    certificate_course_title_encoded = encoders['certificate course title.'].transform([certificate_course_title])[0]

    feature_array = np.array([[gender_encoded, UG_course_encoded, UG_specialization_encoded, interests_encoded,
                               skills_encoded, UG_Percentage, certificate_course_title_encoded]])
    scaled_features = scaler.transform(feature_array)

    probabilities = model.predict_proba(scaled_features)

    top_classes_idx = np.argsort(-probabilities[0])[:3]

    top_classes_names_probs = [(class_names[idx], probabilities[0][idx]) for idx in top_classes_idx]

    return top_classes_names_probs


@app.route("/")
def home():
    return render_template('home.html')


@app.route("/recommend")
def recommend():
    return render_template('recommend.html')

def predict(gender, UG_course, UG_specialization, interests, skills, UG_Percentage, certificate_course_title):
    global model, scaler
    
    time.sleep(2) 
    return [
        ("Computer Software Engineer", 0.65),
        ("Web Developer", 0.17),
        ("Business Analyst", 0.16)
    ]

@app.route('/pred', methods=['POST', 'GET'])
def pred():
    if request.method == 'POST':
        gender = request.form['gender']
        UG_course = request.form['UG_course']
        UG_specialization = request.form['UG_specialization']
        interests = request.form['interests']
        skills = request.form['skills']
        UG_Percentage = request.form['UG_Percentage']
        certificate_course_title = request.form['certificate_course_title']
        try:
            recommendations = Recommendations(gender, UG_course, UG_specialization, interests, skills, UG_Percentage,certificate_course_title)
            return render_template('results.html', recommendations=recommendations)
        except Exception as e:
            app.logger.error(f"Prediction error: {str(e)}")
            return jsonify({'error': 'An unexpected error occurred during prediction'}), 500

    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)

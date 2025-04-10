from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__,template_folder='templates')


# Load the saved model and preprocessor
model = joblib.load('model.pkl')  # Your XGBoost model
preprocessor = joblib.load('preprocessor.pkl')  # Your ColumnTransformer
print(type(preprocessor))
@app.route('/')
def home():
    return render_template('index.html')
    #return "<h1>Hello, Flask is working!</h1>"
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        features = {
            'CreditScore': float(request.form['credit_score']),
            'Age': float(request.form['age']),
            'Tenure_x': float(request.form['tenure']),
            'Balance': float(request.form['balance']),
            'NumOfProducts': float(request.form['num_products']),
            'HasCrCard': request.form['has_credit_card'],
            'IsActiveMember': request.form['is_active'],
            'EstimatedSalary': float(request.form['estimated_salary']),
            'Geography': request.form['geography'],
            'Gender': request.form['gender']
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([features])
        print(input_df.isnull().sum())
        print(input_df.dtypes)
        # Apply preprocessing
        print("DEBUG >>> Type of preprocessor before transform:", type(preprocessor))
        processed_features = preprocessor.transform(input_df)
        
        print(processed_features)
        
        # Make prediction
        prediction = model.predict(processed_features)[0]
        
        result = {
            'churn_prediction': 'Customer likely to churn' if prediction == 1 else 'Customer likely to stay'
        }
        
        return render_template('result.html', result=result)
    
    except Exception as e:
        return render_template('result.html', error=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
model_path = 'model/model.pkl'
with open(model_path, 'rb') as file:
    stacking_clf = pickle.load(file)

# Load your training data for scaling (ensure the path is correct)
training_data_path = r'C:\Users\haris\OneDrive\Desktop\Projects\Data Science mini project\flask\clean.csv'  # Adjust this path
training_data = pd.read_csv(training_data_path)

# Extract the features used for training
features = training_data[['dti', 'annual income', 'revol.bal', 'revol.util', 
                           'days.with.cr.line', 'fico', 'pub.rec', 'inq.last.6mths']]

# Create a StandardScaler instance and fit it on training data
scaler = StandardScaler()
scaler.fit(features)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/loan_prediction', methods=['GET', 'POST'])
def loan_prediction():
    if request.method == 'POST':
        user_input = [
            float(request.form['dti']),
            float(request.form['annual_income']),
            float(request.form['revol_bal']),
            float(request.form['revol_util']),
            float(request.form['days_with_cr_line']),
            float(request.form['fico']),
            int(request.form['pub_rec']),
            int(request.form['inq_last_6mths'])
        ]
        
        # Scale user input
        user_input_scaled = scaler.transform(np.array([user_input]))
        
        # Make prediction
        prediction = stacking_clf.predict(user_input_scaled)
        result = "Approved" if prediction[0] == 1 else "Rejected"
        return render_template('loan_prediction.html', result=result)

    return render_template('loan_prediction.html', result=None)

@app.route('/loan_valuation', methods=['GET', 'POST'])
def loan_valuation():
    if request.method == 'POST':
        annual_salary = float(request.form['annual_salary'])
        credit_score = float(request.form['credit_score'])
        outstanding_loan_amount = float(request.form['outstanding_loan_amount'])
        
        recommended_loan_amount = calculate_loan_amount(annual_salary, credit_score, outstanding_loan_amount)
        return render_template('loan_valuation.html', recommended_loan_amount=recommended_loan_amount)

    return render_template('loan_valuation.html', recommended_loan_amount=None)

@app.route('/risk_assessment', methods=['GET', 'POST'])
def risk_assessment():
    if request.method == 'POST':
        user_data = {
            'credit.policy': int(request.form['credit_policy']),
            'int_rate': float(request.form['int_rate']),
            'installment': float(request.form['installment']),
            'log.annual.inc': float(request.form['log_annual_inc']),
            'dti': float(request.form['dti']),
            'fico': int(request.form['fico']),
            'days.with.cr.line': float(request.form['days_with_cr_line']),
            'revol.bal': float(request.form['revol_bal']),
            'revol.util': float(request.form['revol_util']),
            'inq.last.6mths': int(request.form['inq_last_6mths']),
            'delinq.2yrs': int(request.form['delinq_2yrs']),
            'pub.rec': int(request.form['pub_rec'])
        }

        risk_score, risk_level = calculate_risk(user_data)
        return render_template('risk_assessment.html', risk_score=risk_score, risk_level=risk_level)

    return render_template('risk_assessment.html', risk_score=None)

def calculate_loan_amount(annual_salary, credit_score, outstanding_loan_amount):
    base_loan_amount = annual_salary * 0.3
    if 600 <= credit_score < 700:
        credit_score_factor = 0.8
    elif 700 <= credit_score < 800:
        credit_score_factor = 1.0
    elif credit_score >= 800:
        credit_score_factor = 1.2
    else:
        credit_score_factor = 0.5
    adjusted_loan_amount = base_loan_amount * credit_score_factor
    final_loan_amount = adjusted_loan_amount - outstanding_loan_amount
    return max(final_loan_amount, 0)

def calculate_risk(user_data):
    risk_score = 0

    # Credit Policy
    if user_data['credit.policy'] == 1:
        risk_score -= 1  # Lower risk if credit policy is met
    else:
        risk_score += 1  # Higher risk if credit policy is not met

    # FICO Score
    fico = user_data['fico']
    if fico >= 700:
        risk_score -= 1  # Lower risk
    elif fico >= 650:
        risk_score += 0  # Moderate risk
    else:
        risk_score += 1  # Higher risk

    # Debt-to-Income Ratio (DTI)
    dti = user_data['dti']
    if dti < 30:
        risk_score -= 1  # Lower risk
    elif dti <= 40:
        risk_score += 0  # Moderate risk
    else:
        risk_score += 1  # Higher risk

    # Interest Rate
    int_rate = user_data['int_rate']
    if int_rate < 5:  # Assuming interest rate is in percentage
        risk_score -= 1  # Lower risk
    elif int_rate <= 10:
        risk_score += 0  # Moderate risk
    else:
        risk_score += 1  # Higher risk

    # Days with Credit Line
    days_with_cr_line = user_data['days.with.cr.line']
    if days_with_cr_line > 365:
        risk_score -= 1  # Lower risk
    elif days_with_cr_line >= 180:
        risk_score += 0  # Moderate risk
    else:
        risk_score += 1  # Higher risk

    # Revolving Utilization
    revol_util = user_data['revol.util']
    if revol_util < 30:  # Assuming this is a percentage
        risk_score -= 1  # Lower risk
    elif revol_util <= 50:
        risk_score += 0  # Moderate risk
    else:
        risk_score += 1  # Higher risk

    # Inquiries in Last 6 Months
    inquiries = user_data['inq.last.6mths']
    if inquiries == 0:
        risk_score -= 1  # Lower risk
    elif inquiries <= 2:
        risk_score += 0
    else:
        risk_score += 1  # Higher risk

    # Clamp the risk score between 0 and 5
    risk_score = max(0, min(risk_score, 5))
    risk_level = ""

    if risk_score <= 2:
        risk_level = "Low Risk"
    elif risk_score <= 4:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    return risk_score, risk_level

if __name__ == '__main__':
    app.run(debug=True)

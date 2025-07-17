import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('best_model_xgb.pkl')
scaler = joblib.load('scaler.pkl')

def predict_from_csv(input_csv):
    df = pd.read_csv(input_csv)
    df['Scaled_Amount'] = scaler.transform(df['Amount'].values.reshape(-1, 1))
    df = df.drop(['Amount', 'Time'], axis=1)
    predictions = model.predict(df.drop('Class', axis=1))
    probabilities = model.predict_proba(df.drop('Class', axis=1))[:, 1]
    df['Prediction'] = predictions
    df['Probability'] = probabilities
    df.to_csv('predicted_output.csv', index=False)
    print('Prediction complete. Saved to predicted_output.csv')

# Example: predict_from_csv('new_data.csv')

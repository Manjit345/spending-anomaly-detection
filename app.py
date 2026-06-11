import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib

scaler = joblib.load('scaler.pkl')

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(6, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, 3),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 6),
            nn.ReLU(),
            nn.Linear(6, 12),
            nn.ReLU(),
            nn.Linear(12, 6)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
model = AutoEncoder()
model.load_state_dict(torch.load('autoencoder.pth',map_location='cpu'))
model.eval()

THRESHOLD = 0.057

st.title('Personal Spending Anomaly Detector')
st.write('Enter your spending data(CSV format) to check for anomalies.')

st.subheader('Expected Format')
st.info("""
Your CSV must contain the following columns with exact naming:

| Column | Description |
|--------|-------------|
| `type` | Transaction type which either must be `CASH_OUT` or `TRANSFER` |
| `amount` | Transaction amount in local currency |
| `oldbalanceOrg` | Sender's account balance before transaction |
| `newbalanceOrig` | Sender's account balance after transaction |
| `oldbalanceDest` | Receiver's account balance before transaction |
| `newbalanceDest` | Receiver's account balance after transaction |

Extra columns are ignored. Column names are case-sensitive.  
You can also download the template below, fill it manually and upload it if you want.
""")

template_df = pd.DataFrame(columns=['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'])

st.download_button(
    label='Download CSV Template',
    data=template_df.to_csv(index=False),
    file_name='transaction_template.csv',
    mime='text/csv'
)

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # Define required columns
    required_columns = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

    # Validate only critical columns
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    # Keep only the columns the model needs
    df = df[required_columns]
    
    # Preprocess
    df = df[df['type'].isin(['CASH_OUT', 'TRANSFER'])]
    df = df.drop(columns=['step', 'nameOrig', 'nameDest', 'isFlaggedFraud', 'isFraud'], errors='ignore')
    df['type'] = df['type'].map({'CASH_OUT': 0, 'TRANSFER': 1})
    
    # Scale and convert
    scaled = scaler.transform(df)
    tensor = torch.tensor(scaled.astype(np.float32))
    
    # Get reconstruction errors
    with torch.no_grad():
        outputs = model(tensor)
    errors = torch.mean((tensor - outputs)**2, dim=1).numpy()
    
    # Flag anomalies
    df['reconstruction_error'] = errors
    df['anomaly'] = (errors > THRESHOLD).astype(int)
    
    flagged = df[df['anomaly'] == 1]
    
    st.subheader('Summary')
    st.write(f'Total transactions analysed: {len(df)}')
    st.write(f'Total anomalies flagged: {len(flagged)}')
    
    st.subheader('Flagged Transactions')
    st.dataframe(flagged)
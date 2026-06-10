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

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(f'Uploaded {len(df)} transactions.')
    
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
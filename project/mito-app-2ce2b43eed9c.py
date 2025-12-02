import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor as xr
from sklearn.preprocessing import PolynomialFeatures as pf
import joblib

import warnings
warnings.filterwarnings('ignore')

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        .stAppDeployButton {display:none;}
        footer {visibility: hidden;}
        .stMainBlockContainer {padding: 2rem 1rem 2rem 1rem;}
    </style>
""", unsafe_allow_html=True)

st.title('Tehran House Price Prediction Analysis')

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('tehranhouses.csv')
    return df

df = load_data()

st.header('Data Loading and Preprocessing')

st.subheader('Data Cleaning Steps')
st.markdown("""
The following preprocessing steps were applied to the dataset:
- Changed Parking, Warehouse, and Elevator columns to integer type
- Filtered out problematic Area values
- Cleaned Area column by removing commas and converting to integer
- Sorted by Price and Area
- Filtered Price > 60,000,000,000
- Removed Price(USD) column
""")

# Data preprocessing
from mitosheet.public.v3 import *

# Changed Parking to dtype int
df['Parking'] = df['Parking'].astype('int64')

# Changed Warehouse to dtype int
df['Warehouse'] = df['Warehouse'].astype('int64')

# Changed Elevator to dtype int
df['Elevator'] = df['Elevator'].astype('int64')

# First filter out problematic values from Area before converting to int
# Filter Area
df = df[df['Area'].apply(lambda val: all(val != s for s in ['3,310,000,000', '2,550,000,000', '16,160,000,000', '1,000', '8,400,000,000', '3,600']))]

# Clean and convert Area to int by removing commas first
df['Area'] = df['Area'].str.replace(',', '').astype('int64')

# Sorted Price in ascending order
df = df.sort_values(by='Price', ascending=True, na_position='first')

# Sorted Price in descending order
df = df.sort_values(by='Price', ascending=False, na_position='last')

# Sorted Area in ascending order
df = df.sort_values(by='Area', ascending=True, na_position='first')

# Sorted Price in descending order
df = df.sort_values(by='Price', ascending=False, na_position='last')

# Filtered Price
df = df[df['Price'] > 60000000000]

# Deleted columns Price(USD)
df.drop(['Price(USD)'], axis=1, inplace=True)

st.success(f'Data loaded successfully! Shape: {df.shape}')
st.dataframe(df.head())

st.header('Exploratory Data Analysis')

st.subheader('Area vs Price Visualization')

# Construct the graph and style it
fig = px.bar(df, x='Area', y='Price')
fig.update_layout(
    title='Area, Price bar chart', 
    xaxis={
        "showgrid": True, 
        "rangeslider": {
            "visible": True, 
            "thickness": 0.05
        }
    }, 
    yaxis={
        "showgrid": True
    }, 
    legend={
        "orientation": 'v'
    }, 
    barmode='group', 
    paper_bgcolor='#FFFFFF'
)
st.plotly_chart(fig, use_container_width=True)

st.header('Feature Engineering')

st.subheader('One-Hot Encoding Address Column')

address_dummy = pd.get_dummies(df['Address']).astype('int64')
df_final = df.merge(address_dummy, left_index=True, right_index=True)
df_final.drop(columns='Address', inplace=True)

st.write('First 3 rows after encoding:')
st.dataframe(df_final.head(3))

st.header('Model Training')

st.subheader('Train-Test Split')

x = df_final.drop(columns='Price', axis=1)
y = df_final['Price']

st.write('Features (X):')
st.dataframe(x.head())

st.write('Target (Y):')
st.dataframe(y.head())

# Configure test size
test_size = st.slider('Test Set Size', min_value=0.1, max_value=0.9, value=0.5, step=0.1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

st.write(f'Training set size: {x_train.shape[0]} samples')
st.write(f'Test set size: {x_test.shape[0]} samples')

st.dataframe(x_train.head())

st.subheader('Random Forest Regressor')

# Model parameters
col1, col2, col3 = st.columns(3)
with col1:
    n_estimators = st.number_input('Number of Estimators', min_value=100, max_value=1000, value=500, step=100)
with col2:
    max_depth = st.number_input('Max Depth', min_value=5, max_value=50, value=20, step=5)
with col3:
    min_samples_split = st.number_input('Min Samples Split', min_value=2, max_value=10, value=4, step=1)

if st.button('Train Model'):
    with st.spinner('Training Random Forest model...'):
        model = rf(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(x_train, y_train)
        
        st.success('Model trained successfully!')
        
        st.header('Model Evaluation')
        
        yhat = model.predict(x_test)
        
        mae = mean_absolute_error(y_test, yhat)
        r2 = r2_score(y_test, yhat)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric('Mean Absolute Error', f'{mae:,.0f}')
        with col2:
            st.metric('R² Score', f'{r2:.4f}')
        
        st.subheader('Prediction vs Actual')
        
        comparison_df = pd.DataFrame({
            'Actual': y_test.values,
            'Predicted': yhat
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(comparison_df['Actual'], comparison_df['Predicted'], alpha=0.5)
        ax.plot([comparison_df['Actual'].min(), comparison_df['Actual'].max()], 
                [comparison_df['Actual'].min(), comparison_df['Actual'].max()], 
                'r--', lw=2)
        ax.set_xlabel('Actual Price')
        ax.set_ylabel('Predicted Price')
        ax.set_title('Actual vs Predicted House Prices')
        st.pyplot(fig)
        
        st.dataframe(comparison_df.head(10))

st.markdown('---')
st.markdown('**Analysis Complete**')

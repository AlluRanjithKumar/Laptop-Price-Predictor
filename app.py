import streamlit as st
import pickle
import numpy as np
import pandas as pd

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.slider('Scrensize in inches', 10.0, 18.0, 13.0)

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',df['Cpu brand'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['Gpu brand'].unique())

os = st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size    

    input_df = {
    'Company': [company],
    'TypeName': [type],
    'Ram': [ram],
    'Weight': [weight],
    'Touchscreen': [touchscreen],
    'Ips': [ips],
    'ppi': [ppi],
    'Cpu brand': [cpu],
    'HDD': [hdd],
    'SSD': [ssd],
    'Gpu brand': [gpu],
    'os': [os]
    }

    query_df = pd.DataFrame(input_df)

    # Ensure numeric columns are correct type
    query_df['Touchscreen'] = query_df['Touchscreen'].astype(int)
    query_df['Ips'] = query_df['Ips'].astype(int)
    query_df['Ram'] = query_df['Ram'].astype(int)
    query_df['Weight'] = query_df['Weight'].astype(float)
    query_df['ppi'] = query_df['ppi'].astype(float)
    query_df['HDD'] = query_df['HDD'].astype(int)
    query_df['SSD'] = query_df['SSD'].astype(int)

    # Predict
    prediction = np.exp(pipe.predict(query_df)[0])
    st.title(f"The predicted price of this configuration is ₹ {int(prediction)}")


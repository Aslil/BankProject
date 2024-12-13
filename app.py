import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

# Streamlit başlığı
st.title('📊 Bank Subscription Prediction App 📊')

# Kullanıcıdan giriş alınması
age = st.number_input('Please enter your age:', step=1)

job = st.selectbox(
    'Please select your job:',
    ('admin', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired',
     'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown')
)

marital = st.selectbox('Please select your marital status:', ('married', 'single', 'divorced'))

education = st.selectbox('Please select your education level:',
                          ('basic', 'high school', 'illiterate',
                           'professional course', 'university degree', 'unknown'))

housing = st.selectbox('Select if you have a housing loan', ('yes', 'no', 'unknown'))

loan = st.selectbox('Select if you have a personal loan?', ('yes', 'no', 'unknown'))

day_of_week = st.selectbox('Select your last contact day of the week:',
                           ('mon', 'tue', 'wed', 'thu', 'fri'))

contact = st.selectbox('Select your contact communication type:', ('cellular', 'telephone'))

month = st.selectbox('Select your last contact month:',
                      ('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'))

duration = st.number_input('Enter your last contact duration in seconds:', step=1)

campaign = st.number_input('Enter the number of calls made during the campaign:', step=1)

cons_price_idx = st.number_input('Enter the consumer price index:')

cons_conf_idx = st.number_input('Enter the consumer confidence index:')

euribor3m = st.number_input('Enter Euribor 3-month rate:')

nr_employed = st.number_input('Enter the number of employees:')

# Kullanıcıdan alınan girdilerle bir DataFrame oluşturma
input_data = pd.DataFrame({
    'age': [age],
    'job': [job],
    'marital': [marital],
    'education': [education],
    'housing': [housing],
    'loan': [loan],
    'day_of_week': [day_of_week],
    'contact': [contact],
    'month': [month],
    'duration': [duration],
    'campaign': [campaign],
    'cons.price.idx': [cons_price_idx],
    'cons.conf.idx': [cons_conf_idx],
    'euribor3m': [euribor3m],
    'nr.employed': [nr_employed]
})

# Model yükleme
model_path = 'tuned_best_model.pkl'
try:
    model = pickle.load(open(model_path, 'rb'))
except FileNotFoundError:
    st.error('Model file not found. Please check the correct path.')
    st.stop()

# Tahmin yapma
if st.button('Make Prediction'):
    try:
        # Tahmin yapılmadan önce veriyi ve çıktıyı yazdır
        st.write("Input data:")
        st.write(input_data)  # Kullanıcıdan alınan veriyi ekranda gösterir
        
        # Model tahmini
        prediction = model.predict(input_data)
        
        # Tahmin edilen sonucu yazdır
        st.write("Model Output:")
        st.write(prediction[0])
        
        # Kullanıcıya sonucu göster
        if prediction[0] == 'yes':
            st.success('Yes! The client subscribes to a time deposit.')
        else:
            st.error('No! The client does not subscribe to a time deposit.')
    except Exception as e:
        st.error(f'An error occurred during prediction: {e}')

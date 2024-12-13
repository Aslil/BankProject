import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

# Streamlit başlığı
st.title('📊 Banka Pazarlama Tahmin Uygulaması')

# Kullanıcıdan giriş alınması
age = st.number_input('Lütfen yaşınızı girin:', step=1)

job = st.selectbox(
    'Lütfen mesleğinizi seçin:',
    ('admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired',
     'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown')
)

marital = st.selectbox('Medeni durumunuzu seçin:', ('married', 'single', 'divorced'))

education = st.selectbox('Eğitim durumunuzu seçin:',
                          ('basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate',
                           'professional.course', 'university.degree', 'unknown'))

housing = st.selectbox('Konut krediniz var mı?', ('yes', 'no', 'unknown'))

loan = st.selectbox('Kişisel krediniz var mı?', ('yes', 'no', 'unknown'))

day_of_week = st.selectbox('Son iletişim gününüzü seçin:',
                           ('mon', 'tue', 'wed', 'thu', 'fri'))

contact = st.selectbox('İletişim türünü seçin:', ('cellular', 'telephone'))

month = st.selectbox('Son iletişim ayını seçin:',
                      ('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'))

duration = st.number_input('Son görüşme süresini (saniye) girin:', step=1)

campaign = st.number_input('Kampanya sırasında yapılan arama sayısı:', step=1)

cons_price_idx = st.number_input('Tüketici fiyat endeksini girin:')

cons_conf_idx = st.number_input('Tüketici güven endeksini girin:')

euribor3m = st.number_input('Euribor 3 aylık oranını girin:')

nr_employed = st.number_input('Çalışan sayısını girin:')

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
    st.error('Model dosyası bulunamadı. Lütfen doğru yolu kontrol edin.')
    st.stop()

# Tahmin yapma
if st.button('Tahmin Yap'):
    try:
        # Tahmin yapılmadan önce veriyi ve çıktıyı yazdır
        st.write("Giriş verisi:")
        st.write(input_data)  # Kullanıcıdan alınan veriyi ekranda gösterir
        
        # Model tahmini
        prediction = model.predict(input_data)
        
        # Tahmin edilen sonucu yazdır
        st.write("Model Çıktısı:")
        st.write(prediction[0])
        
        # Kullanıcıya sonucu göster
        if prediction[0] == 'yes':
            st.success('Evet! Müşteri bir vadeli mevduata abone olur.')
        else:
            st.error('Hayır! Müşteri bir vadeli mevduata abone olmaz.')
    except Exception as e:
        st.error(f'Tahmin sırasında bir hata oluştu: {e}')

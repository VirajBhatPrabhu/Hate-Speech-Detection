import streamlit as st
import pickle

model=pickle.load(open('Hatespeechmodel.pkl','rb'))
cv=pickle.load(open('countvectorizer.pkl','rb'))


st.title("Hate Speech Recognition")
user = st.text_input("Enter a Tweet For Recognition: ")
if st.button('Detect'):
        sample = user
        data = cv.transform([sample]).toarray()
        a = model.predict(data)
        st.title(a)
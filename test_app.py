import streamlit as st

st.set_page_config(page_title="Streamlit Test")

st.title("Hello Sneha!")
st.success("Streamlit is rendering correctly.")

name = st.text_input("Enter your name")

if name:
    st.write(f"Welcome, {name}!")

st.button("Test Button")

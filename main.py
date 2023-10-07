import textwrap
import streamlit as st

from yt_assist.yt_assist import get_vectordb_from_url, get_response_from_query

st.title("Youtube Assistant")
url = st.text_input("Paste video URL here: ", max_chars=50)
query = st.text_input("Ask me about the video: ", max_chars=50)
button = st.button("GO")

if button:
    db = None
    if len(url):
        db = get_vectordb_from_url(url)

    response = ""
    st.subheader("Answer: ")
    if db and len(query):
        response = get_response_from_query(query, db)
        st.text(
            textwrap.fill(response, width=80)
        )
    else:
        st.write("You forgot to ask me a question!")

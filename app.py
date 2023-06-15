import streamlit as st
from app_functions import get_title_channel
from app_functions import get_video_id
from app_functions import create_database
from app_functions import get_response

st.title('Youtube Video Chatbot 💻🤖💬')
st.markdown(f'<p style="text-align: left; font-size: 20px;"><strong>Engage in a conversation with an AI model about any YouTube video by simply entering its URL.</strong></p>', unsafe_allow_html=True)
video_url = st.text_input('Enter the video URL')

if video_url:
    video_id = get_video_id(video_url)
    title, channel_title = get_title_channel(video_id)
    st.markdown(f'<p style="text-align: left; font-size: 20px;"><strong>Video Title: {title}</strong></p>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: left; font-size: 20px;"><strong>Channel Name: {channel_title}</strong></p>', unsafe_allow_html=True)

    prompt = st.text_input('Enter Prompt') 
    if prompt:
        db = create_database(video_url)
        response = get_response(db, prompt)
        st.write(response)
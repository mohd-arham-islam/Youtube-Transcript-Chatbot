import streamlit as st
import requests
import json
import os
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI

API_KEY = "AIzaSyAu8S7E4EzMJKP0g2NgoG5NY_KWydfdmho"
OPENAI_API_KEY = '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'

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

embeddings = OpenAIEmbeddings()

def get_video_id(video_url):
    url = video_url
    index = url.find('watch?v=')
    video_id = url[index+8:]
    return video_id

def get_title_channel(video_id):
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet&id={video_id}&fields=items(id%2Csnippet)&key={API_KEY}"
    response = requests.get(url)
    data = json.loads(response.text)
    title = data["items"][0]["snippet"]["title"]
    channel_title = data["items"][0]["snippet"]["channelTitle"]
    return title, channel_title

def create_database(video_url: str): 
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response(db, query, k=4):
    
    """
    Setting the chunksize to 1000 and k to 4 maximizes the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model_name="text-davinci-003")

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Consider the following prompt: {question}
        And answer by searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I'm sorry, I couldn't find this information in the transcript".
        
        Your answers should be verbose and detailed.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response

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
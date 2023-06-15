import requests
import json
import os
from api import API_KEY, OPENAI_API_KEY
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI

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
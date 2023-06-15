# Youtube-Transcript-Chatbot
## An application that allows users to engage in a conversation with an AI model about any YouTube video by simply entering its URL.

The application utilizes LangChain to provide OpenAI's LLM (text-davinci-003) with access to the transcript of the entered YouTube video. To obtain the video transcript, I utilized the ***YoutubeLoader*** class from the ***langchain.document_loaders*** module.

To optimize the process, I employed FAISS, a library developed by Meta AI, to create a vector database that enables functions like similarity search. By leveraging this approach, the LLM no longer needs to analyze the entire transcript. Instead, it focuses only on the sections that are similar to the user's input prompt. As a result, this significantly reduces the time and computational resources required for processing.

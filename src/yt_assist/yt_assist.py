from dotenv import load_dotenv

from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS, VectorStore

load_dotenv()

EMBEDDINGS = OpenAIEmbeddings()

def get_vectordb_from_youtube(url: str) -> VectorStore:
    # Get transcript from video_url
    yt_loader = YoutubeLoader.from_youtube_url(url)
    transcript = loader.load()

    # Split transcript into documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(transcripts)

    # Create vector db from transript using OpenAI embeddings
    db = FAISS.from_documents(docs,
        embeddings=EMBEDDINGS)

    return db
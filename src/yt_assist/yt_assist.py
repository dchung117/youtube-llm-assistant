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
TEMPLATE="""
    You are a helpful assistant that that can answer questions about youtube videos
    based on the video's transcript.

    Answer the following question: {question}
    By searching the following video transcript: {docs}

    Only use the factual information from the transcript to answer the question.

    If you feel like you don't have enough information to answer the question, say "I don't know".

    Your answers should be verbose and detailed.
"""

def get_vectordb_from_url(url: str) -> VectorStore:
    """
    Get VectorStore database of Youtube transcript from video url.

    Args
    ----
        url: str
            Youtube video url
    Return
    ------
        VectorStore
            Vector database of embedded transcript documents.
    """
    # Get transcript from video_url
    yt_loader = YoutubeLoader.from_youtube_url(url)
    transcript = yt_loader.load()

    # Split transcript into documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(transcript)

    # Create vector db from documents using OpenAI embeddings
    db = FAISS.from_documents(docs,
        embedding=EMBEDDINGS)

    return db

def get_response_from_query(query: str,
    db: VectorStore,
    k: int = 4) -> str:
    # Get similar documents to query
    docs = db.similarity_search(query, k=k)
    docs_content = " ".join([d.page_content for d in docs])

    # Create chain
    llm = OpenAI(model="text-davinci-003")
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template=TEMPLATE
    )
    chain = LLMChain(
        llm=llm,
        prompt=prompt
    )

    # Get response from formatted prompt
    response = chain(
        {"question": query, "docs": docs_content}
    )
    response = response.replace("\n", "")
    return response
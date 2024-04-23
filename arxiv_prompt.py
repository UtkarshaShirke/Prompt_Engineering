import os
import time
import requests
import xml.etree.ElementTree as ET
import streamlit as st
#from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec, PodSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import langchain_community
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_community.vectorstores import Pinecone
import langchain
import pinecone
import toml




# Load environment variables
#load_dotenv()

# Pinecone Setup
api_key = st.secrets['PINECONE_API_KEY']
environment = st.secrets['PINECONE_ENVIRONMENT']
use_serverless = os.environ.get("USE_SERVERLESS", "False").lower() == "true"

# Configure Pinecone client
pc = pinecone.Pinecone(api_key=api_key, environment=environment)
spec = ServerlessSpec(cloud='gcp-starter', region='us-central1') if use_serverless else PodSpec(environment=environment)

# Define or choose your index name
index_name = 'arxiv-papers'

def initialize_chain(selected_paper):
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)

    pc.create_index(
        index_name,
        dimension=1536,
        metric='cosine',
        spec=spec
    )

    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

    loader = PyPDFLoader(selected_paper['pdf_link'])
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200,length_function=len)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    # docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
    docsearch = langchain_community.vectorstores.Pinecone.from_documents(docs, embeddings, index_name=index_name)

    llm = ChatOpenAI(model_name='gpt-4')
    retriever = docsearch.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.session_state.chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, max_tokens_limit=4000)
    st.session_state.chat_active = True  # Enable chat interface

def fetch_papers(query, max_results=25):
    base_url = "http://export.arxiv.org/api/query?"
    query_params = f"search_query={query}&start=0&max_results={max_results}"
    response = requests.get(base_url + query_params)
    feed = ET.fromstring(response.content)

    papers = []
    for entry in feed.findall('{http://www.w3.org/2005/Atom}entry'):
        paper_id = entry.find('{http://www.w3.org/2005/Atom}id').text.split('/abs/')[-1]
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
        pdf_link = f"https://arxiv.org/pdf/{paper_id}.pdf"
        papers.append({'id': paper_id, 'title': title, 'summary': summary, 'pdf_link': pdf_link})
    
    return papers

def is_response_relevant(question, response, embeddings):
    # Generate embeddings for both the question and the response
    question_embedding = embeddings.get_embeddings(question)
    response_embedding = embeddings.get_embeddings(response)

    # Calculate cosine similarity between the question and the response embeddings
    if np.any(question_embedding) and np.any(response_embedding):
        similarity = cosine_similarity([question_embedding], [response_embedding])[0][0]
    else:
        similarity = 0  # Handle cases where embeddings are empty or invalid

    # Define a threshold for determining relevance
    threshold = 0.5  # Adjust this based on your testing and what makes sense for your use case

    return similarity > threshold

def chat():
    if 'chain' in st.session_state and st.session_state.chat_active:
        current_paper_id = st.session_state.current_paper_id
        if current_paper_id:
            # Ensure there's a chat history for the current paper
            if current_paper_id not in st.session_state.paper_chat_histories:
                st.session_state.paper_chat_histories[current_paper_id] = []
            
            chat_history = st.session_state.paper_chat_histories[current_paper_id]

            # Apply the CSS styling
            st.markdown(css, unsafe_allow_html=True)

            # Display chat messages with HTML template
            chat_messages = ''
            for message in chat_history:
                if message['role'] == 'user':
                    chat_messages += user_template.replace('{{MSG}}', message['content'])
                else:
                    chat_messages += bot_template.replace('{{MSG}}', message['content'])

            st.markdown(chat_messages, unsafe_allow_html=True)

            # Handle text input and update chat history
            def handle_text_input():
                user_input = st.session_state.chat_input.strip()  # Ensure whitespace isn't causing issues
                if user_input:
                    chat_history.append({'role': 'user', 'content': user_input})

                    # Compose the conversation into a single string for processing
                    conversation = [f"{item['role']}: {item['content']}" for item in chat_history]
                    # max_context_length = 4096  # Adjust based on the model's maximum token limit
                    # combined_conversation = "\n".join(conversation)[-max_context_length:]

                    # Get response from the conversational model
                    response = st.session_state.chain.run({'question': user_input})

                    # Append model response to chat history
                    chat_history.append({'role': 'assistant', 'content': response})

                    # Clear the input box after processing
                    st.session_state.chat_input = ''

                    # Optionally, truncate chat history to manage token size
                    # Keep only the last 10 exchanges
                    if len(chat_history) > 20:
                        st.session_state.paper_chat_histories[st.session_state.current_paper_id] = chat_history[-20:]
                    
                                
            # Set up user input at the bottom. It will submit on Enter key press.
            st.text_input("Ask me about the paper:", key="chat_input", on_change=handle_text_input, placeholder="Type your question and press Enter")

      
st.title('ArXiv Research Paper Bot')


# Initialize chat messages and chat history for each paper
if 'paper_chat_histories' not in st.session_state:
    st.session_state.paper_chat_histories = {}

if 'current_paper_id' not in st.session_state:
    st.session_state.current_paper_id = None

# Sidebar tab selection
sidebar_tab = st.sidebar.radio("Menu", ["Search Papers", "Index Papers"])

if sidebar_tab == "Search Papers":
    # Reset chat state when returning to the search page
    st.session_state.paper_chat_histories = {}
    st.session_state.current_paper_id = None
    st.session_state.chat_active = False  # Disable the chat interface

    query = st.sidebar.text_input('Enter your search query:', '')
    fetch_button = st.sidebar.button('Fetch Papers')
    if fetch_button and query:
        st.session_state.papers = fetch_papers(query)
        for paper in st.session_state.papers:
            with st.expander(f"{paper['title']}"):
                st.write(f"**Summary:** {paper['summary']}")
                st.markdown(f"[Read Full Paper PDF]({paper['pdf_link']})", unsafe_allow_html=True)


elif sidebar_tab == "Index Papers":
    if 'papers' in st.session_state and st.session_state.papers:
        paper_titles = [paper['title'] for paper in st.session_state.papers]
        selected_paper_title = st.sidebar.selectbox('Select a paper to index:', paper_titles)
        index_button = st.sidebar.button('Index Selected Paper')
        if index_button and selected_paper_title:
            selected_paper = next((paper for paper in st.session_state.papers if paper['title'] == selected_paper_title), None)
            if selected_paper:
                st.session_state.current_paper_id = selected_paper['id']
                initialize_chain(selected_paper)  # Initialize chat for new paper
                st.success("Welcome to Arxiv Research Paper Bot. You can now ask questions related to your research paper.")

chat()

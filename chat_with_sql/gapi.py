# Import required modules
import streamlit as st
from pathlib import Path
from langchain.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler

# Load environment variables
from dotenv import load_dotenv
import os
load_dotenv()  # Load variables from .env

# Import Google LLM (Palm 2 or Gemini)
from langchain_google_genai import ChatGoogleGenerativeAI

# Set the Streamlit page config
st.set_page_config(page_title="Chat with SQL DB", page_icon='💻')
st.title('Chat with your SQL DB Dynamically!!')

# Constants for database options
LOCAL_DB = 'USE_LOCALDB'
MYSQL = 'USE_MYSQL'

# Sidebar radio to choose DB type
radio_opt = ['Use SQLite 3 Database - prreport.db', 'Connect to your database']
selected_opt = st.sidebar.radio(label='Choose the DB which you want to chat with', options=radio_opt)

# Handle MySQL DB input
if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input('Provide Host Name')
    mysql_user = st.sidebar.text_input('MySQL Username')
    mysql_pass = st.sidebar.text_input('Provide the Password', type='password')
    mysql_db = st.sidebar.text_input('MySQL Database Name')
else:
    db_uri = LOCAL_DB

# Load Google API Key from .env
api_key = os.getenv("GOOGLE_API_KEY")

# Ensure the API key exists
if not api_key:
    st.error('Please add GOOGLE_API_KEY in your .env file')
    st.stop()

# Initialize Google LLM (like Palm2 / Gemini) from Langchain
llm = ChatGoogleGenerativeAI(
    model="models/chat-bison-001",  # or "gemini-pro" if available
    google_api_key=api_key,
    temperature=0.2,
    streaming=True
)

# Function to configure and return the SQL database engine
@st.cache_resource(ttl='2h')
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_pass=None, mysql_db=None):
    if db_uri == LOCAL_DB:
        # Load SQLite database from project directory
        dbfilepath = (Path(__file__).parent / 'pr_report.db').absolute()
        return SQLDatabase(create_engine(f'sqlite:///{dbfilepath}'))
    
    elif db_uri == MYSQL:
        # Check for valid input before connecting to MySQL
        if not (mysql_host and mysql_user and mysql_pass and mysql_db):
            st.error('Please provide the MySQL database information')
            st.stop()

        # Return MySQL DB engine
        return SQLDatabase(
            create_engine(f'mysql+mysqlconnector://{mysql_user}:{mysql_pass}@{mysql_host}/{mysql_db}')
        )

# Initialize the database connection
if db_uri == MYSQL:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_pass, mysql_db)
else:
    db = configure_db(db_uri)

# Setup Langchain SQL Toolkit using the database and LLM
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Create a Langchain SQL agent
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    handle_parsing_errors=True,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# Manage chat history in session state
if 'messages' not in st.session_state or st.sidebar.button('clear message history'):
    st.session_state['messages'] = [{'role': 'assistant', 'content': 'How can I help you?'}]

# Display the previous chat messages
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# Input box for user to ask SQL questions
user_query = st.chat_input(placeholder='Ask anything from the database')

# If user sends a message
if user_query:
    # Add user message to chat history
    st.session_state.messages.append({'role': 'user', 'content': user_query})
    st.chat_message('user').write(user_query)

    # Show assistant response container
    with st.chat_message('assistant'):
        streamlit_callback = StreamlitCallbackHandler(st.container())

        # Run query with SQL agent
        response = agent.run(user_query, callbacks=[streamlit_callback])

        # Append response to history and show in chat
        st.session_state.messages.append({'role': 'assistant', 'content': response})
        st.write(response)

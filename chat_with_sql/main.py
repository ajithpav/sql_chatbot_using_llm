# Import necessary modules
import streamlit as st  # Streamlit for the frontend UI
from pathlib import Path  # For working with file paths
from langchain.sql_database import SQLDatabase  # Langchain SQL DB interface
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit  # Toolkit for SQL agent
from sqlalchemy import create_engine  # SQLAlchemy engine for DB connection
import sqlite3  # SQLite native Python driver
from langchain_groq import ChatGroq  # Groq integration for Langchain
from langchain.agents import create_sql_agent  # Function to create an SQL agent
from langchain.agents.agent_types import AgentType  # Defines agent behavior
from langchain.callbacks import StreamlitCallbackHandler  # For real-time responses in Streamlit

# Set the page title and favicon
st.set_page_config(page_title="Chat with SQL DB", page_icon='💻')

# Main app title
st.title('Chat with your SQL DB Dynamically!!')

# Constants to identify DB types
LOCAL_DB = 'USE_LOCALDB'
MYSQL = 'USE_MYSQL'

# Sidebar option to select between SQLite or MySQL
radio_opt = ['Use SQLite 3 Database - prreport.db', 'Connect to your database']
selected_opt = st.sidebar.radio(label='Choose the DB which you want to chat with', options=radio_opt)

# If MySQL is selected, show DB credentials inputs
if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input('Provide Host Name')  # MySQL host
    mysql_user = st.sidebar.text_input('MySQL Username')  # MySQL username
    mysql_pass = st.sidebar.text_input('Provide the Password', type='password')  # MySQL password (hidden)
    mysql_db = st.sidebar.text_input('MySQL Database Name')  # MySQL DB name
else:
    db_uri = LOCAL_DB  # Default to local SQLite DB

# Sidebar input for Groq API key to use Llama3
api_key = st.sidebar.text_input(label='Groq API Key', type='password')

# Validation: Ensure Groq API key is provided
if not api_key:
    st.info('Please add the Groq API Key')
    st.stop()  # Stop execution if key is missing

# Initialize Groq's Llama3 model
llm = ChatGroq(
    groq_api_key=api_key,
    model_name='Llama3-8b-8192',
    streaming=True  # Enable real-time streaming output
)

# Function to set up database connection (cached for 2 hours)
@st.cache_resource(ttl='2h')
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_pass=None, mysql_db=None):
    if db_uri == LOCAL_DB:
        # Resolve full path to SQLite file
        dbfilepath = (Path(__file__).parent / 'pr_report.db').absolute()
        return SQLDatabase(create_engine(f'sqlite:///{dbfilepath}'))  # Return SQLite DB engine
    
    elif db_uri == MYSQL:
        # Validate MySQL credentials
        if not (mysql_host and mysql_user and mysql_pass and mysql_db):
            st.error('Please provide the MySQL database information')
            st.stop()

        # Create MySQL engine using mysqlconnector
        return SQLDatabase(
            create_engine(f'mysql+mysqlconnector://{mysql_user}:{mysql_pass}@{mysql_host}/{mysql_db}')
        )

# Get the database object based on user selection
if db_uri == MYSQL:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_pass, mysql_db)
else:
    db = configure_db(db_uri)

# Initialize the Langchain SQL toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Create the SQL agent
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    handle_parsing_errors=True,  # If parsing fails, try to recover
    verbose=True,  # Log outputs for debugging
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION  # Uses description-based zero-shot prompting
)

# Initialize or reset chat history
if 'messages' not in st.session_state or st.sidebar.button('clear message history'):
    st.session_state['messages'] = [{'role': 'assistant', 'content': 'How can I help you?'}]

# Display all previous messages in the chat interface
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# Chat input box for user query
user_query = st.chat_input(placeholder='Ask anything from the database')

# When user submits a query
if user_query:
    # Add user message to session state
    st.session_state.messages.append({'role': 'user', 'content': user_query})
    st.chat_message('user').write(user_query)

    # Assistant response section
    with st.chat_message('assistant'):
        streamlit_callback = StreamlitCallbackHandler(st.container())  # Enables streaming response

        # Run the agent with user query and get the response
        response = agent.run(user_query, callbacks=[streamlit_callback])

        # Save assistant response in session state
        st.session_state.messages.append({'role': 'assistant', 'content': response})
        st.write(response)  # Display the response on UI

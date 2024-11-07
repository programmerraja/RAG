import logging
import re
from enum import Enum
from io import StringIO

import psycopg2
import requests
import streamlit as st

from langchain.text_splitter import (
    MarkdownTextSplitter,
    CharacterTextSplitter,
    NLTKTextSplitter,
)

from ollama import Client
from youtube_transcript_api import YouTubeTranscriptApi

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkingMethods(Enum):
    CHARACTER_SPLITER = "CHARACTER"
    MARKDOWN_SPLITER = "MARKDOWN"
    TOKEN_SPLITER = "TOKEN"


class RAG:
    def __init__(
        self,
        chunk_size=2000,
        chunk_overlap=20,
        chunking_method=ChunkingMethods.MARKDOWN_SPLITER.value,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_method = chunking_method

    def chunking(self, data):
        print(self.chunking_method)
        if self.chunking_method == ChunkingMethods.MARKDOWN_SPLITER.value:
            text_splitter = MarkdownTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            return text_splitter.create_documents([data])
        if self.chunking_method == ChunkingMethods.CHARACTER_SPLITER.value:
            text_splitter = CharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            return text_splitter.create_documents([data])
        if self.chunking_method == ChunkingMethods.TOKEN_SPLITER.value:
            text_splitter = NLTKTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            return text_splitter.create_documents([data])

    def pre_retrieval_optimization(self, query):
        pass


class PGDatabase:
    def __init__(
        self,
        table_name,
        db_url,
        ollama=None,
        ollama_host=None,
        openai_key=None,
        embedding_model="nomic-embed-text",
        completion_model="tinyllama",
        rag=None,
    ):
        self.conn = self.connect_db(db_url)
        self.table_name = table_name
        self.embedding_table_name = f"embedding_{table_name}"
        self.is_db_running_local = False
        self.ollama = ollama
        self.ollama_host = ollama_host
        self.openai_key = openai_key
        self.embedding_model = embedding_model
        self.completion_model = completion_model
        self.rag = rag

        if ollama:
            self.set_ollama_config()

    def connect_db(self, db_url):
        try:
            logger.info(f"Connecting to database with URL: {db_url}")
            conn = psycopg2.connect(db_url)
            logger.info("Database connection established")
            return conn
        except psycopg2.DatabaseError as e:
            logger.error(f"Database connection error: {e}")
            raise

    def set_ollama_config(self):
        try:
            logger.info("Setting Ollama configuration")
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT set_config('ai.ollama_host', %s, false);",
                    (self.ollama_host,),
                )
                self.conn.commit()
            logger.info("Ollama configuration set successfully")
        except psycopg2.DatabaseError as e:
            logger.error(f"Error setting Ollama config: {e}")

    def create_table(self):
        try:
            logger.info(f"Creating table {self.table_name} if not exists")
            with self.conn.cursor() as cur:
                if self.openai_key:
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self.table_name} (
                            id SERIAL PRIMARY KEY,
                            title TEXT,
                            content TEXT,
                            uniqueid TEXT,
                        );
                    """)
                else:
                    cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id SERIAL PRIMARY KEY,
                        title TEXT,
                        content TEXT,
                        uniqueid TEXT,
                        embedding VECTOR(768)
                    );
                """)
                self.conn.commit()
            logger.info(f"Table {self.table_name} created successfully")
        except psycopg2.DatabaseError as e:
            logger.error(f"Error creating table: {e}")

    def insert_data(self, data):
        try:
            logger.info(f"Inserting data into table {self.table_name}")

            with self.conn.cursor() as cur:
                cur.execute(
                    f"SELECT 1 FROM {self.table_name} WHERE uniqueid = %s",
                    (data["uniqueid"],),
                )
                if cur.fetchone() is not None:
                    logger.info(
                        f"Data with uniqueid {data['uniqueid']} already exists. Skipping insertion."
                    )
                    return

                if not self.openai_key:
                    docs = self.rag.chunking(data["content"])

                    logger.info(f"Split content into {len(docs)} chunks")

                    for doc in docs:
                        data["embedding"] = self.ollama.embeddings(
                            model=self.embedding_model, prompt=doc.page_content
                        )["embedding"]
                        cur.execute(
                            f"""
                            INSERT INTO {self.table_name} (title, uniqueid, content, embedding) 
                            VALUES (%s, %s, %s, %s)
                        """,
                            (
                                data["title"],
                                data["uniqueid"],
                                data["content"],
                                data["embedding"],
                            ),
                        )
                else:
                    cur.execute(
                        f"""
                            INSERT INTO {self.table_name} (title, uniqueid, content) 
                            VALUES (%s, %s, %s)
                        """,
                        (
                            data["title"],
                            data["uniqueid"],
                            data["content"],
                        ),
                    )
                    cur.execute(f"""
                        SELECT ai.create_vectorizer(
                            '{self.table_name}'::regclass,
                            destination => '{self.embedding_table_name}',
                            embedding => ai.embedding_openai('{self.embedding_model}', 768),
                            chunking => ai.chunking_recursive_character_text_splitter('content')
                        );
                    """)
                self.conn.commit()
                logger.info("Data inserted successfully")
        except psycopg2.DatabaseError as e:
            logger.error(f"Error inserting data: {e}")

    def fetch_data(self):
        try:
            logger.info(f"Fetching data from table {self.table_name}")
            with self.conn.cursor() as cur:
                cur.execute(
                    f"SELECT title, content, vector_dims(embedding) FROM {self.table_name};"
                )
                rows = cur.fetchall()
                for row in rows:
                    logger.info(
                        f"Title: {row[0]}, Content: {row[1]}, Embedding Dimensions: {row[2]}"
                    )
        except psycopg2.DatabaseError as e:
            logger.error(f"Error fetching data: {e}")

    def retrieve_and_generate_response(self, query, limit=2):
        try:
            logger.info(f"Retrieving and generating response for query: {query}")

            with self.conn.cursor() as cur:
                query_embedding = self.get_query_embedding(query, cur)
                cur.execute(
                    f"""
                    SELECT title, content, 1 - (embedding <=> %s::vector) AS similarity
                    FROM {self.table_name}
                    ORDER BY similarity DESC
                    LIMIT %s;
                """,
                    (query_embedding, limit),
                )

                rows = cur.fetchall()

                logger.info(f"Top {limit} similar rows: {rows}")

                context = "\n\n".join(
                    [f"Title: {row[0]}\n Content: {row[1]}" for row in rows]
                )
                response = self.generate_response(query, context, cur)

                logger.info(f"Generated response: {response}")
                return response
        except psycopg2.DatabaseError as e:
            logger.error(f"Error retrieving and generating response: {e}")

    def get_query_embedding(self, query, cur):
        if not self.is_db_running_local:
            return self.ollama.embeddings(model=self.embedding_model, prompt=query)[
                "embedding"
            ]
        cur.execute(f"SELECT ai.ollama_embed('{self.embedding_model}', %s);", (query,))
        return cur.fetchone()

    def generate_response(self, query, context, cur):
        prompt = f"Query: {query} \n Context: {context} \n NOTE: !! Only return answer for the query !!"

        if self.is_db_running_local:
            cur.execute(
                f"SELECT ai.ollama_generate('{self.completion_model}', %s);", (prompt,)
            )
            return cur.fetchone()["response"]

        return self.ollama.chat(
            model=self.completion_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a friendly chatbot that always answers questions based on the provided context. If the context is insufficient to answer a question, please respond by saying, 'I don't have enough context to answer this question.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )["message"]["content"]

    def clean_table(self):
        try:
            logger.info(
                f"Cleaning table {self.table_name} and {self.embedding_table_name}"
            )
            with self.conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {self.table_name};")
                if self.openai_key:
                    cur.execute(f"TRUNCATE TABLE {self.embedding_table_name};")
                self.conn.commit()
            logger.info("Tables cleaned successfully")
        except psycopg2.DatabaseError as e:
            logger.error(f"Error cleaning table: {e}")


class PGVector:
    def __init__(
        self,
        pg_database,
    ):
        self.db = pg_database
        self.db.create_table()

    def fetch_data_from_website(self, url):
        try:
            logger.info(f"Fetching data from website: {url}")
            response = requests.get(
                f"https://r.jina.ai/{url}",
                headers={
                    "X-With-Links-Summary": "false",
                    "X-Remove-Selector": "a,image",
                    "X-With-Images-Summary": "false",
                    "Accept": "application/json",
                },
            )
            response.raise_for_status()

            response = response.json()

            if response and response["code"] == 200 and response["data"]:
                logger.info("Data fetched successfully from website")
                return {
                    "title": response["data"]["title"],
                    "content": response["data"]["content"],
                    "uniqueid": response["data"]["url"],
                }
        except requests.RequestException as e:
            logger.error(f"Error fetching data from website: {e}")
        return None

    def fetch_transcript_from_youtube(self, url):
        try:
            video_id = re.search(r"v=([^&]+)", url).group(1)
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join([entry["text"] for entry in transcript])
            logger.info("Transcript fetched successfully from YouTube")
            return {
                "title": f"YouTube Video {video_id}",
                "content": transcript_text,
                "uniqueid": url,
            }
        except Exception as e:
            logger.error(f"Error fetching transcript from YouTube: {e}")
        return None


class ChatApp:
    def __init__(self):
        self.db_url = st.sidebar.text_input("Enter PGVector DB URL")
        
        self.chat_mode = st.sidebar.selectbox(
            "Select Chat Mode", ["Chat with Website", "Chat with File"]
        )
        self.llm_choice = st.sidebar.selectbox("Select LLM", ["OpenAI", "OLLAMA"])

        self.embedding_model = self.get_embedding_model()
        self.completion_model = self.get_completion_model()

        self.openai_key = None

        self.ollama_host = None
        self.ollama = None
      

        st.session_state.isWebsiteAdded = (
            False
            if "isWebsiteAdded" not in st.session_state
            else st.session_state.get("isWebsiteAdded")
        )

        self.isWebsiteAdded = st.session_state.get("isWebsiteAdded")

        if self.llm_choice == "OpenAI":
            self.openai_key = st.sidebar.text_input("Enter OpenAI Key", type="password")
        elif self.llm_choice == "OLLAMA":
            self.ollama_host =  st.sidebar.text_input("Enter OLLAMA Host URL")
            self.ollama = Client(host=self.ollama_host)
            if "chunk_size" not in st.session_state:
                st.session_state.chunk_size = 2000
            if "chunk_overlap" not in st.session_state:
                st.session_state.chunk_overlap = 20
            if "chunking_method" not in st.session_state:
                st.session_state.chunking_method = (
                    ChunkingMethods.MARKDOWN_SPLITER.value
                )

            self.chunk_size = st.sidebar.number_input(
                "Set chunk size",
                min_value=100,
                value=st.session_state.chunk_size,
                max_value=10000,
                on_change=lambda: st.session_state.update(
                    {"chunk_size": self.chunk_size}
                ),
            )
            st.session_state.chunk_size = self.chunk_size

            self.chunk_overlap = st.sidebar.number_input(
                "Set chunk overlap",
                min_value=0,
                value=st.session_state.chunk_overlap,
                max_value=100,
                on_change=lambda: st.session_state.update(
                    {"chunk_overlap": self.chunk_overlap}
                ),
            )

            st.session_state.chunk_overlap = self.chunk_overlap

            self.chunking_method = st.sidebar.selectbox(
                "Select chunking method",
                [
                    ChunkingMethods.MARKDOWN_SPLITER.value,
                    ChunkingMethods.CHARACTER_SPLITER.value,
                    ChunkingMethods.TOKEN_SPLITER.value,
                ],
                index=[
                    ChunkingMethods.MARKDOWN_SPLITER.value,
                    ChunkingMethods.CHARACTER_SPLITER.value,
                    ChunkingMethods.TOKEN_SPLITER.value,
                ].index(st.session_state.chunking_method),
                on_change=lambda: st.session_state.update(
                    {"chunking_method": self.chunking_method}
                ),
            )
            st.session_state.chunking_method = self.chunking_method

        if "reterival_limit" not in st.session_state:
            st.session_state.reterival_limit = 3

        self.reterival_limit = st.sidebar.number_input(
            "Set number of doc to retrieve",
            min_value=1,
            value=st.session_state.reterival_limit,
            max_value=10,
        )
        self.initialize_session_state()

        # Add a button to clear all data in the database
        if st.sidebar.button("Clear All Data IN DB"):
            self.clear_all_data()

    def get_embedding_model(self):
        if self.llm_choice == "OpenAI":
            return st.sidebar.selectbox(
                "Select Embedding Model",
                ["text-embedding-3-small", "text-embedding-3-large"],
            )
        elif self.llm_choice == "OLLAMA":
            return st.sidebar.selectbox("Select Embedding Model", ["nomic-embed-text"])

    def get_completion_model(self):
        if self.llm_choice == "OLLAMA":
            return st.sidebar.selectbox(
                "Select Completion Model",
                ["tinyllama", "llama3", "mistral", "phi3.5"],
            )

    def initialize_session_state(self):
        if "chat_with_website_history" not in st.session_state:
            st.session_state.chat_with_website_history = [
                {
                    "role": "assistant",
                    "content": "Please provide a website link to start the chat with it.",
                }
            ]
        if "chat_with_file_history" not in st.session_state:
            st.session_state.chat_with_file_history = []

    def check_llm_config(self):
        logger.info("Checking LLM configuration")
        if self.llm_choice == "OpenAI" and not self.openai_key:
            st.error("Please enter the OpenAI key")
            return False
        elif self.llm_choice == "OLLAMA" and not self.ollama_host:
            st.error("Please enter the OLLAMA host URL")
            return False
        elif not self.db_url:
            st.error("Please enter the PGVector DB URL")
            return False
        logger.info("LLM configuration is valid")
        return True

    def clear_all_data(self):
        if "pg_vector" in st.session_state:
            st.session_state.pg_vector.db.clean_table()

        st.session_state.chat_with_website_history = [
            {
                "role": "assistant",
                "content": "Please provide a website link to start the chat with it.",
            }
        ]
        st.session_state.chat_with_file_history = []
        st.session_state.isWebsiteAdded = False
        st.success("All data cleared successfully.")

    def handle_chat_with_website(self):
        st.title("Chat with website")
        url_pattern = re.compile(r"^(https?://)")
        youtube_pattern = re.compile(r"(https?://)?(www\.)?(youtube|youtu\.be)")

        logger.info("Handling chat with website")

        for msg in st.session_state.chat_with_website_history:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input():
            st.session_state.chat_with_website_history.append(
                {"role": "user", "content": prompt}
            )
            st.chat_message("user").write(prompt)

            if not self.check_llm_config():
                return

            if "pg_vector" not in st.session_state:
                pg_database = PGDatabase(
                    table_name="chat_with_website",
                    db_url=self.db_url,
                    ollama=self.ollama,
                    ollama_host=self.ollama_host,
                    openai_key=self.openai_key,
                    embedding_model=self.embedding_model,
                    completion_model=self.completion_model,
                    rag=RAG(self.chunk_size, self.chunk_overlap, self.chunking_method),
                )
                st.session_state.pg_vector = PGVector(pg_database)
            pg_vector = st.session_state.pg_vector

            if prompt == "clear":
                pg_vector.db.clean_table()
                st.session_state.chat_with_website_history = []
                st.session_state.isWebsiteAdded = False
                return

            if url_pattern.match(prompt):
                if youtube_pattern.match(prompt):
                    with st.spinner("Fetching YouTube transcript..."):
                        logger.info(f"Fetching YouTube transcript for URL: {prompt}")
                        youtube_content = pg_vector.fetch_transcript_from_youtube(
                            prompt
                        )
                        if youtube_content:
                            pg_vector.db.insert_data(youtube_content)
                            st.chat_message("assistant").write(
                                "Transcript fetched and indexed. You can now start chatting."
                            )
                            st.session_state.isWebsiteAdded = True
                        else:
                            st.chat_message("assistant").write(
                                "Unable to fetch the transcript."
                            )
                        return
                else:
                    with st.spinner("Indexing the website content..."):
                        logger.info(f"Indexing website content for URL: {prompt}")
                        website_content = pg_vector.fetch_data_from_website(prompt)
                        if website_content:
                            pg_vector.db.insert_data(website_content)
                            st.chat_message("assistant").write(
                                "Indexing completed. You can now start chatting."
                            )
                            st.session_state.isWebsiteAdded = True
                        else:
                            st.chat_message("assistant").write(
                                "Invalid URL or unable to fetch content."
                            )
                        return
            # print(st.session_state.isWebsiteAdded, "isWebsiteAdded")

            if st.session_state.isWebsiteAdded:
                with st.spinner("Assistant is generating a response..."):
                    logger.info(f"Generating response for prompt: {prompt}")

                    response = pg_vector.db.retrieve_and_generate_response(
                        prompt, self.reterival_limit
                    )
            else:
                response = "Please provide a website link with (https://.. or http://) to start the chat with it."

            st.session_state.chat_with_website_history.append(
                {"role": "assistant", "content": response}
            )

            st.chat_message("assistant").write(response)

    def handle_chat_with_file(self):
        st.title("Chat with File")
        st.write("Upload a file to begin a conversation based on its content.")

        uploaded_file = st.file_uploader("Choose a file", type=["txt", "md"])

        if uploaded_file is not None:
            logger.info(f"File uploaded: {uploaded_file.name} {uploaded_file.type}")

            file_content = StringIO(uploaded_file.getvalue().decode("utf-8")).read()

            st.write("File content successfully loaded!")
            st.session_state.file_content = file_content

            if not self.check_llm_config():
                return

            if "pg_vector" not in st.session_state:
                pg_database = PGDatabase(
                    table_name="chat_with_file",
                    db_url=self.db_url,
                    ollama=self.ollama,
                    ollama_host=self.ollama_host,
                    openai_key=self.openai_key,
                    embedding_model=self.embedding_model,
                    completion_model=self.completion_model,
                    rag=RAG(self.chunk_size, self.chunk_overlap, self.chunking_method),
                )
                st.session_state.pg_vector = PGVector(pg_database)
            pg_vector = st.session_state.pg_vector
            # print(file_content, "file_content")
            pg_vector.db.insert_data(
                {
                    "content": file_content,
                    "title": uploaded_file.name,
                    "uniqueid": uploaded_file.name,
                }
            )
            logger.info("File content inserted into database")

            if "chat_with_file_history" not in st.session_state:
                st.session_state["chat_with_file_history"] = []

            for msg in st.session_state.chat_with_file_history:
                st.chat_message(msg["role"]).write(msg["content"])

            if prompt := st.chat_input():
                st.session_state.chat_with_file_history.append(
                    {"role": "user", "content": prompt}
                )
                st.chat_message("user").write(prompt)

                if prompt == "clear":
                    logger.info("Clearing chat history and database tables")
                    pg_vector.db.clean_table()
                    st.session_state.chat_with_file_history = []
                    return

                with st.spinner("Assistant is generating a response..."):
                    response = pg_vector.db.retrieve_and_generate_response(
                        prompt, self.reterival_limit
                    )

                st.session_state.chat_with_file_history.append(
                    {"role": "assistant", "content": response}
                )
                st.chat_message("assistant").write(response)
        else:
            st.write("Please upload a file to start the conversation.")
            logger.warning("No file uploaded for chat")

    def run(self):
        if self.chat_mode == "Chat with Website":
            self.handle_chat_with_website()
        elif self.chat_mode == "Chat with File":
            self.handle_chat_with_file()


if __name__ == "__main__":
    chat_app = ChatApp()
    chat_app.run()

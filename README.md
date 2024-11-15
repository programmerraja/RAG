# RAG

## Overview
RAG (Retrieval-Augmented Generation) is a comprehensive UI application designed to explore the capabilities of RAG models. It supports various interaction modes and is built for the Dev.to hackathon.

## Features
- **Chat with Website**: Enter a website URL  or youtube url to chat with its content.
- **Chat with File**: Upload a text or markdown file to initiate a conversation based on its content.
- **Chat with Dev.to Articles**: Provide a Dev.to username to chat with the user's articles.
- **Multiple LLM Support**: Choose between OpenAI and OLLAMA for language model interactions.
- **Customizable Chunking**: Configure chunk size, overlap, and method for text processing.
- **Database Integration**: Utilize PGVector with PostgreSQL for data storage and retrieval.

## Getting Started
1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/rag.git
    cd rag
    ```

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the application**:
    ```sh
    streamlit run rag.py
    ```

## Configuration
- **OpenAI Key**: Input your OpenAI API key in the sidebar if using OpenAI.
- **OLLAMA Host URL**: Provide the OLLAMA host URL if using OLLAMA.
- **PGVector DB URL**: Enter the PostgreSQL database URL for PGVector.

## Usage
- **Chat with Website**: Select "Chat with Website" mode, input a website URL, and start chatting.
- **Chat with File**: Select "Chat with File" mode, upload a file, and begin the conversation.
- **Chat with Dev.to Articles**: Select "Chat with your Dev.to articles" mode, enter a Dev.to username, and start chatting.

## License
This project is licensed under the MIT License.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.



## Misc

to install psycopg2 on ubuntu 

if you issue refer https://stackoverflow.com/questions/11618898/pg-config-executable-not-found

sudo apt-get install libpq-dev python-dev

pip install psycopg2-binary


sudo OLLAMA_HOST="0.0.0.0" ollama serve



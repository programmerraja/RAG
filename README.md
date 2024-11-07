# RAG

## Overview
RAG (Retrieval-Augmented Generation) is a simple and intuitive UI interface designed to experiment with and explore the capabilities of RAG models. This project was built for the Dev.to hackathon.

## Features
- **Chat with Website**: Provide a website link and chat with the content of the website.
- **Chat with File**: Upload a file and have a conversation based on its content.
- **Multiple LLM Support**: Choose between OpenAI and OLLAMA for language model interactions.
- **Customizable Chunking**: Set chunk size, overlap, and chunking method for text processing.
- **Database Integration**: Store and retrieve data using PGVector with PostgreSQL.

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
- **OpenAI Key**: Enter your OpenAI API key in the sidebar.
- **OLLAMA Host URL**: Provide the OLLAMA host URL if using OLLAMA.
- **PGVector DB URL**: Enter the PostgreSQL database URL for PGVector.

## Usage
- **Chat with Website**: Select "Chat with Website" mode, enter a website URL, and start chatting.
- **Chat with File**: Select "Chat with File" mode, upload a file, and begin the conversation.

## License
This project is licensed under the MIT License.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## Contact
For any questions or feedback, please contact [your email].

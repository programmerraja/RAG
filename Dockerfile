FROM python:3.12-slim

RUN apt-get update && apt-get install libpq-dev python-dev -y && pip install psycopg2-binary

RUN git clone https://github.com/programmerraja/RAG.git .

RUN pip3 install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "rag.py","--server.enableCORS fals", "--server.port=8501", "--server.address=0.0.0.0","--server.enableXsrfProtection false"]
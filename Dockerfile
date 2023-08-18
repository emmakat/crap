FROM continuumio/miniconda3
RUN conda config --add channels conda-forge && conda config --add channels pytorch && conda config --add channels anaconda
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install specified versions of packages available on Conda
RUN conda install -y \
    gensim==4.3.1 \
    numpy==1.25.2 \
    wordcloud==1.9.2 \
    textblob==0.17.1 \
    nltk==3.8.1 \
    streamlit==1.25.0 \
    spacy==3.6.0 \
    seaborn==0.12.2 \
    joblib==1.3.1 \
    pandas==2.0.3 \
    matplotlib==3.7.2 \
    scikit-learn==1.3.0 \
    pillow==9.5.0 \
    altair

# Install en_core_web_sm model from conda-forge channel
RUN conda install -c conda-forge -y spacy-model-en_core_web_sm

# Install remaining packages using pip
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# RUN python3 app.py

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Use uma imagem base do Python multi-arquitetura com Debian Bullseye (Debian 11)
FROM python:3.9-slim-bullseye

# Define variáveis de ambiente
ENV PYTHONUNBUFFERED 1 \
    PIP_NO_CACHE_DIR 1

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Instala ferramentas de compilação para llama-cpp-python e outras libs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    cmake \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    libgomp1 \ 
    && rm -rf /var/lib/apt/lists/*
# A linha 24 (`&& rm -rf...`) é a continuação do comando RUN da linha 16.
# Para que ela seja uma continuação, a linha anterior (23) precisa ter `\` no final.
ENV PYTHONPATH="/workspaces/match_nlp_app:${PYTHONPATH}"

# Copia o arquivo de dependências
COPY requirements.txt .

# Instala as dependências Python
# AJUSTE CHAVE AQUI: Adicionar -pthread ou -lpthread nos CMAKE_ARGS
RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DCMAKE_CXX_FLAGS='-pthread'" \
    pip install --upgrade --no-cache-dir -r requirements.txt

# Copia o restante do código da sua aplicação (src/, data/, notebooks/, Streamlit app)
COPY . .

# Cria um diretório para os modelos GPT4all dentro do contêiner.
# RUN mkdir -p /app/models

# Expõe a porta padrão do Streamlit
EXPOSE 8501

# Comando padrão para iniciar o Streamlit.
# CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
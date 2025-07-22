
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st
import os
import pickle
import hashlib  # Adicionado para gerar chaves de cache únicas

# Importa SentenceTransformer para embeddings de alta qualidade
from sentence_transformers import SentenceTransformer

# A instância LLM do chat_llm.py (usada para gerar explicações, não para embeddings aqui)
from src.chat_llm import ask_llm

# Pasta onde Parquets e embeddings são salvos
PROCESSED_DATA_PATH = "data/processed_data"
VAGA_EMBEDDINGS_FILE = os.path.join(PROCESSED_DATA_PATH, "vaga_embeddings.pkl")
CANDID_EMBEDDINGS_FILE = os.path.join(
    PROCESSED_DATA_PATH, "candid_embeddings.pkl")
PROSPECT_EMBEDDINGS_FILE = os.path.join(
    PROCESSED_DATA_PATH, "prospect_embeddings.pkl")

# NOVO: Caminho para o arquivo de cache das explicações do LLM
LLM_EXPLANATIONS_CACHE_FILE = os.path.join(
    PROCESSED_DATA_PATH, "llm_explanations_cache.pkl")


# --- Modelo de Embedding dedicado para inferência (não para geração em massa aqui) ---
@st.cache_resource(show_spinner="Carregando modelo de embedding (SentenceTransformer) para inferência...")
def load_embedding_model_for_inference():
    """
    Carrega um modelo de embedding pré-treinado (e.g., SentenceTransformer).
    Esta função é executada apenas uma vez devido ao cache do Streamlit.
    Será usada para gerar embeddings de queries pontuais, se necessário.
    """
    print("DEBUG_EMBED: Carregando modelo de embedding SentenceTransformer 'all-MiniLM-L6-v2' para inferência.")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("DEBUG_EMBED: Modelo de embedding carregado com sucesso para inferência.")
        return model
    except Exception as e:
        print(
            f"DEBUG_EMBED: ERRO ao carregar o modelo de embedding para inferência: {e}")
        st.error(
            f"Erro ao carregar o modelo de embedding para inferência: {e}. Verifique a conexão ou os requisitos.")
        st.stop()  # Impede a continuação se o modelo não carregar


# Carrega o modelo de embedding apenas uma vez
embedding_model_inference = load_embedding_model_for_inference()


# --- Funções de Carregamento de Embeddings (Assumem que já foram gerados) ---

@st.cache_data(show_spinner="Carregando embeddings pré-gerados...", persist=True)
def load_all_embeddings():
    """
    Carrega embeddings de arquivos .pkl. Esta função ASSUME que os embeddings
    já foram gerados pelo script 'generate_preprocessed_data.py'.
    """
    embeddings_data = {}
    files_to_load = {
        'jobs': VAGA_EMBEDDINGS_FILE,
        'applicants': CANDID_EMBEDDINGS_FILE,
        'prospects': PROSPECT_EMBEDDINGS_FILE
    }

    for key, file_path in files_to_load.items():
        if not os.path.exists(file_path):
            st.error(
                f"ERRO: Arquivo de embeddings '{file_path}' não encontrado! Por favor, execute 'python scripts/generate_preprocessed_data.py' primeiro.")
            st.stop()  # Parar a aplicação se um arquivo essencial não for encontrado

        try:
            with open(file_path, 'rb') as f:
                embeddings_data[key] = pickle.load(f)
            print(
                f"DEBUG_EMBED: Embeddings para '{key}' carregados de '{file_path}'.")
        except Exception as e:
            st.error(
                f"Erro ao carregar embeddings de '{file_path}': {e}. Tente regenerá-los.")
            st.stop()  # Parar em caso de erro grave de leitura

    return embeddings_data


# --- Funções de Cache para Explicações do LLM ---

def load_llm_explanations_cache():
    """Carrega o dicionário de explicações do LLM de um arquivo pickle."""
    if os.path.exists(LLM_EXPLANATIONS_CACHE_FILE):
        try:
            with open(LLM_EXPLANATIONS_CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
            print(
                f"DEBUG_LLM_CACHE: Cache de explicações do LLM carregado de {LLM_EXPLANATIONS_CACHE_FILE}.")
            return cache
        except Exception as e:
            print(
                f"DEBUG_LLM_CACHE: Erro ao carregar cache de explicações do LLM: {e}. Iniciando cache vazio.")
            return {}
    print("DEBUG_LLM_CACHE: Cache de explicações do LLM não encontrado. Iniciando cache vazio.")
    return {}


def save_llm_explanations_cache(cache):
    """Salva o dicionário de explicações do LLM em um arquivo pickle."""
    try:
        # Garante que o diretório exista
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        with open(LLM_EXPLANATIONS_CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
        print(
            f"DEBUG_LLM_CACHE: Cache de explicações do LLM salvo em {LLM_EXPLANATIONS_CACHE_FILE}.")
    except Exception as e:
        print(
            f"DEBUG_LLM_CACHE: Erro ao salvar cache de explicações do LLM: {e}.")


# --- Funções de Matching ---

def find_top_matches(query_embedding: np.ndarray, target_embeddings_data: dict, top_n: int = 5):
    """
    Encontra os top N itens mais compatíveis para um embedding de consulta.
    `target_embeddings_data` deve ser um dicionário com 'ids' e 'embeddings'.
    """
    target_ids = target_embeddings_data['ids']
    target_embeddings_array = target_embeddings_data['embeddings']

    if not target_embeddings_array.shape[0] > 0:
        print("DEBUG_MATCH: Nenhum embedding alvo para comparar.")
        return pd.DataFrame()

    query_embedding_reshaped = query_embedding.reshape(1, -1)

    # Calcula a similaridade de cosseno entre o embedding da query e todos os embeddings alvo
    similarities = cosine_similarity(
        query_embedding_reshaped, target_embeddings_array)[0]

    # Cria um DataFrame para fácil ordenação e mapeamento de IDs
    match_df = pd.DataFrame({
        'id': target_ids,
        'similarity_score': similarities
    })

    # Ordena e retorna os top N matches
    top_matches = match_df.sort_values(
        by='similarity_score', ascending=False).head(top_n)

    print(f"DEBUG_MATCH: Encontrados {len(top_matches)} top matches.")
    return top_matches

# --- Função de Explicação do LLM para o Match (AGORA COM CACHE) ---


def get_llm_explanation_for_match(job_text: str, candidate_text: str, match_score: float) -> str:
    """
    Usa o LLM para explicar o motivo da seleção de um candidato/prospect para uma vaga.
    Chama a função `ask_llm` do módulo `chat_llm.py` e gerencia um cache persistente.
    """
    # Criar uma chave única para o cache baseada nos inputs relevantes
    # Usamos um hash SHA256 para garantir que a chave seja concisa e única
    # Normalizamos o score para 2 casas decimais para consistência da chave
    key_parts = (job_text, candidate_text, f"{match_score:.2f}")
    cache_key = hashlib.sha256(str(key_parts).encode('utf-8')).hexdigest()

    # Carregar o cache (dentro da função por simplicidade, Streamlit cache faria melhor globalmente)
    llm_explanations_cache = load_llm_explanations_cache()

    if cache_key in llm_explanations_cache:
        print(
            f"DEBUG_LLM_CACHE: Explicação do LLM encontrada no cache para chave: {cache_key}")
        return llm_explanations_cache[cache_key]

    # Se não estiver no cache, gerar a explicação com o LLM
    prompt = f"""Explique em português de forma concisa (máximo 150 palavras) por que o perfil do candidato/prospect descrito abaixo pode ser um bom match para a vaga, considerando uma similaridade de {match_score:.2f} (onde 1.0 é um match perfeito). Foco nos pontos relevantes.

Vaga:
{job_text[:1500]} # Limita o texto da vaga para não exceder o n_ctx do LLM

Perfil do Candidato/Prospect:
{candidate_text[:1500]} # Limita o texto do perfil

Explicação do Match:"""

    print(
        f"DEBUG_LLM_EXPLAIN: Solicitando explicação do LLM (NÃO ESTÁ NO CACHE). Prompt inicial: {prompt[:200]}...")
    # Usa a função `ask_llm` que já tem o spinner e tratamento de erros
    explanation = ask_llm(prompt=prompt, max_tokens=200)
    print(
        f"DEBUG_LLM_EXPLAIN: Explicação do LLM gerada. (Primeiras 50 chars: {explanation[:50]})")

    # Salvar no cache antes de retornar
    llm_explanations_cache[cache_key] = explanation
    save_llm_explanations_cache(llm_explanations_cache)

    return explanation

# Função para gerar embedding de uma única string (para futuras consultas dinâmicas, por exemplo)


def get_single_embedding(text: str) -> np.ndarray:
    """Gera o embedding para uma única string de texto usando o modelo de inferência."""
    if not isinstance(text, str):
        text = str(text)  # Garante que o input é string
    return embedding_model_inference.encode(text, convert_to_numpy=True)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import streamlit as st
# import os
# import pickle  # Para salvar/carregar embeddings

# # Importa SentenceTransformer (adicione 'sentence-transformers' ao requirements.txt)
# from sentence_transformers import SentenceTransformer

# # A instância LLM do chat_llm.py (para explicações)
# from src.chat_llm import ask_llm

# PROCESSED_DATA_PATH = "data/processed_data"
# VAGA_EMBEDDINGS_FILE = os.path.join(PROCESSED_DATA_PATH, "vaga_embeddings.pkl")
# CANDID_EMBEDDINGS_FILE = os.path.join(
#     PROCESSED_DATA_PATH, "candid_embeddings.pkl")
# PROSPECT_EMBEDDINGS_FILE = os.path.join(
#     PROCESSED_DATA_PATH, "prospect_embeddings.pkl")

# # --- Modelo de Embedding dedicado ---
# # O modelo de embedding AGORA É CARREGADO APENAS NO generate_preprocessed_data.py
# # Aqui, apenas o carregamos para uso interno, mas não geramos se não existir.


# @st.cache_resource(show_spinner="Carregando modelo de embedding (SentenceTransformer)...")
# def load_embedding_model_for_inference():  # Renomeado para clareza
#     """Carrega o modelo de embedding para uso em inferência (gerar 1 embedding por vez)."""
#     # Não será usado para geração em massa, apenas para a consulta individual no futuro.
#     # O modelo será baixado na primeira vez que for carregado (se não já baixado pelo script).
#     print("DEBUG_EMBED: Carregando modelo de embedding SentenceTransformer para inferência.")
#     try:
#         model = SentenceTransformer('all-MiniLM-L6-v2')
#         print("DEBUG_EMBED: Modelo de embedding carregado com sucesso para inferência.")
#         return model
#     except Exception as e:
#         print(
#             f"DEBUG_EMBED: ERRO ao carregar o modelo de embedding para inferência: {e}")
#         st.error(
#             f"Erro ao carregar o modelo de embedding para inferência: {e}.")
#         st.stop()


# # Carregar o modelo de embedding apenas para casos de uso de inferência avulsa
# embedding_model_inference = load_embedding_model_for_inference()


# # --- Funções de Carregamento de Embeddings (Assumem que já foram gerados) ---

# @st.cache_data(show_spinner="Carregando embeddings pré-gerados...", persist=True)
# def load_all_embeddings():
#     """
#     Carrega embeddings de arquivos .pkl. Esta função ASSUME que os embeddings
#     já foram gerados pelo script 'generate_preprocessed_data.py'.
#     """
#     embeddings_data = {}
#     files_to_load = {
#         'jobs': VAGA_EMBEDDINGS_FILE,
#         'applicants': CANDID_EMBEDDINGS_FILE,
#         'prospects': PROSPECT_EMBEDDINGS_FILE
#     }

#     for key, file_path in files_to_load.items():
#         if not os.path.exists(file_path):
#             st.error(
#                 f"ERRO: Arquivo de embeddings '{file_path}' não encontrado! Por favor, execute 'python scripts/generate_preprocessed_data.py' primeiro.")
#             st.stop()

#         try:
#             with open(file_path, 'rb') as f:
#                 embeddings_data[key] = pickle.load(f)
#             print(
#                 f"DEBUG_EMBED: Embeddings para '{key}' carregados de '{file_path}'.")
#         except Exception as e:
#             st.error(
#                 f"Erro ao carregar embeddings de '{file_path}': {e}. Tente regenerá-los.")
#             st.stop()

#     return embeddings_data


# # --- Funções de Matching ---

# def find_top_matches(query_embedding: np.ndarray, target_embeddings_data: dict, top_n: int = 5):
#     """Encontra os top N itens mais compatíveis para um embedding de consulta."""
#     target_ids = target_embeddings_data['ids']
#     target_embeddings_array = target_embeddings_data['embeddings']

#     if not target_embeddings_array.shape[0] > 0:
#         print("DEBUG_MATCH: Nenhum embedding alvo para comparar.")
#         return pd.DataFrame()

#     query_embedding_reshaped = query_embedding.reshape(1, -1)

#     similarities = cosine_similarity(
#         query_embedding_reshaped, target_embeddings_array)[0]

#     match_df = pd.DataFrame({
#         'id': target_ids,
#         'similarity_score': similarities
#     })

#     top_matches = match_df.sort_values(
#         by='similarity_score', ascending=False).head(top_n)

#     print(f"DEBUG_MATCH: Encontrados {len(top_matches)} top matches.")
#     return top_matches

# # --- Função de Explicação do LLM (já está em chat_llm.py, mas aqui está a chamada) ---


# def get_llm_explanation_for_match(job_text: str, candidate_text: str, match_score: float) -> str:
#     """Usa o LLM para explicar o motivo da seleção de um candidato/prospect para uma vaga."""
#     prompt = f"""Explique em português de forma concisa (máximo 150 palavras) por que o perfil descrito abaixo é um bom match para a vaga, considerando uma similaridade de {match_score:.2f}.

# Vaga:
# {job_text[:1500]} # Limita o texto para não exceder o n_ctx do LLM e ser conciso

# Perfil do Candidato/Prospect:
# {candidate_text[:1500]} # Limita o texto

# Explicação do Match:"""

#     return ask_llm(prompt=prompt, max_tokens=200)

# # Função para gerar embedding de uma única string (para futuras consultas dinâmicas, por exemplo)


# def get_single_embedding(text: str) -> np.ndarray:
#     """Gera o embedding para uma única string de texto usando o modelo de inferência."""
#     if not isinstance(text, str):
#         text = str(text)  # Garante que o input é string
#     return embedding_model_inference.encode(text, convert_to_numpy=True)

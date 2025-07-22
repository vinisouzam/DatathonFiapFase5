import pandas as pd
import os
import streamlit as st  # Para st.cache_data e exibir mensagens de erro

# Define caminhos absolutos baseados no WORKDIR do Docker (/workspaces/match_nlp_app)
BASE_DATA_PATH = "data"
PROCESSED_DATA_PATH = os.path.join(BASE_DATA_PATH, "processed_data")


@st.cache_data(show_spinner="Carregando dados processados do Parquet...", persist=True)
def load_processed_data():
    """
    Carrega dados dos arquivos Parquet pré-existentes.
    Esta função ASSUME que os arquivos Parquet já foram gerados
    pelo script 'generate_preprocessed_data.py'.
    """
    jobs_parquet_path = os.path.join(PROCESSED_DATA_PATH, "vagas.parquet")
    applicants_parquet_path = os.path.join(
        PROCESSED_DATA_PATH, "applicants.parquet")
    prospects_parquet_path = os.path.join(
        PROCESSED_DATA_PATH, "prospects.parquet")

    # Verifica se TODOS os arquivos Parquet esperados existem
    all_parquet_exist = (
        os.path.exists(jobs_parquet_path) and
        os.path.exists(applicants_parquet_path) and
        os.path.exists(prospects_parquet_path)
    )

    if not all_parquet_exist:
        st.error("ERRO: Arquivos Parquet processados não encontrados! Por favor, execute 'python scripts/generate_preprocessed_data.py' primeiro.")
        st.stop()  # Impede a execução se os arquivos essenciais não existirem

    print(f"DEBUG_DL: Carregando dados do Parquet de: {PROCESSED_DATA_PATH}")
    try:
        jobs_df = pd.read_parquet(jobs_parquet_path)
        applicants_df = pd.read_parquet(applicants_parquet_path)
        prospects_df = pd.read_parquet(prospects_parquet_path)
        # print(prospects_df.columns)
        print(f"DEBUG_DL: Dados do Parquet carregados com sucesso.")
        return jobs_df, applicants_df, prospects_df
    except Exception as e:
        print(
            f"DEBUG_DL: ERRO ao ler arquivos Parquet: {e}. Verifique se foram gerados corretamente.")
        st.error(
            f"Erro ao ler arquivos Parquet: {e}. Por favor, tente re-executar o script de pré-processamento.")
        st.stop()  # Parar em caso de erro de leitura grave

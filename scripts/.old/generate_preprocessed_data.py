# scripts/generate_preprocessed_data.py

import pandas as pd
import os
import json
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Reutiliza funções de extração de texto (pode copiar ou importar se for mantê-las em utils.py)
# Para simplificar e evitar dependências circulares de importação neste script, vou duplicá-las aqui.
# Em um projeto maior, você teria um módulo shared_utils.py.


def extract_text_from_job_data(job_data: dict) -> str:
    profile = job_data.get("perfil_vaga", {})
    activities = profile.get("principais_atividades", "")
    skills = profile.get("competencia_tecnicas_e_comportamentais", "")
    title = job_data.get("informacoes_basicas", {}).get("titulo_vaga", "")
    level = profile.get("nivel profissional", "")
    area = profile.get("areas_atuacao", "")
    return " ".join(filter(None, [title, level, area, activities, skills])).strip()


def extract_text_from_applicant_data(applicant_data: dict) -> str:
    cv_pt = applicant_data.get("cv_pt", "")
    professional_info = applicant_data.get("informacoes_profissionais", {})
    tech_knowledge = professional_info.get("conhecimentos_tecnicos", "")
    return " ".join(filter(None, [cv_pt, tech_knowledge])).strip()


def extract_text_from_prospect_data(prospect_entry: dict, vaga_titulo: str = "") -> str:
    name = prospect_entry.get("nome", "")
    comment = prospect_entry.get("comentario", "")
    situation = prospect_entry.get("situacao_candidado", "")
    text_parts = [vaga_titulo, name, comment, situation]
    return " ".join(filter(None, text_parts)).strip()


# Caminhos para os arquivos
BASE_DATA_PATH = "data"
PROCESSED_DATA_PATH = os.path.join(BASE_DATA_PATH, "processed_data")
VAGA_EMBEDDINGS_FILE = os.path.join(PROCESSED_DATA_PATH, "vaga_embeddings.pkl")
CANDID_EMBEDDINGS_FILE = os.path.join(
    PROCESSED_DATA_PATH, "candid_embeddings.pkl")
PROSPECT_EMBEDDINGS_FILE = os.path.join(
    PROCESSED_DATA_PATH, "prospect_embeddings.pkl")


def run_preprocessing_and_embedding_generation():
    """
    Carrega dados brutos, pré-processa, gera embeddings e os salva.
    """
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    print("--- 1. Carregando e Pré-processando Dados JSON ---")

    # Vagas
    vagas_path = os.path.join(BASE_DATA_PATH, "vagas.json")
    try:
        with open(vagas_path, 'r', encoding='utf-8') as f:
            jobs_raw = json.load(f)
        jobs_df = pd.DataFrame.from_dict(jobs_raw, orient='index')
        jobs_df['id_vaga'] = jobs_df.index
        jobs_df['processed_text'] = jobs_df.apply(
            extract_text_from_job_data, axis=1)
        print(f"  {len(jobs_df)} vagas carregadas e pré-processadas.")
    except Exception as e:
        print(f"  ERRO ao ler/processar vagas.json: {e}")
        jobs_df = pd.DataFrame()

    # Candidatos
    applicants_path = os.path.join(BASE_DATA_PATH, "applicants.json")
    try:
        with open(applicants_path, 'r', encoding='utf-8') as f:
            applicants_raw = json.load(f)
        applicants_df = pd.DataFrame.from_dict(applicants_raw, orient='index')
        applicants_df['id_candidato'] = applicants_df.index
        applicants_df['processed_text'] = applicants_df.apply(
            extract_text_from_applicant_data, axis=1)
        print(f"  {len(applicants_df)} candidatos carregados e pré-processados.")
    except Exception as e:
        print(f"  ERRO ao ler/processar applicants.json: {e}")
        applicants_df = pd.DataFrame()

    # Prospects
    prospects_path = os.path.join(BASE_DATA_PATH, "prospects.json")
    prospects_df = pd.DataFrame()
    if os.path.exists(prospects_path):
        try:
            with open(prospects_path, 'r', encoding='utf-8') as f:
                prospects_raw = json.load(f)

            all_prospects_data = []
            for vaga_id, vaga_content in prospects_raw.items():
                vaga_titulo = vaga_content.get("titulo", "")
                if "prospects" in vaga_content and isinstance(vaga_content["prospects"], list):
                    for p_entry in vaga_content["prospects"]:
                        p_copy = p_entry.copy()
                        p_copy['id_vaga_associada'] = vaga_id
                        p_copy['processed_text'] = extract_text_from_prospect_data(
                            p_copy, vaga_titulo=vaga_titulo)
                        all_prospects_data.append(p_copy)
            prospects_df = pd.DataFrame(all_prospects_data)
            prospects_df['id_prospect'] = range(len(prospects_df))
            if 'codigo' in prospects_df.columns:
                prospects_df = prospects_df.set_index('codigo', drop=False)
                prospects_df.index.name = 'id_prospect'
            else:
                prospects_df = prospects_df.set_index('id_prospect')

            print(f"  {len(prospects_df)} prospects carregados e pré-processados.")
        except Exception as e:
            print(f"  ERRO ao ler/processar prospects.json: {e}")
    else:
        print(f"  Arquivo prospects.json NÃO ENCONTRADO em {prospects_path}")

    # Salvar DataFrames processados em Parquet
    if not jobs_df.empty:
        jobs_df.to_parquet(os.path.join(
            PROCESSED_DATA_PATH, "vagas.parquet"), index=True)
        print(
            f"  Vagas salvas em {os.path.join(PROCESSED_DATA_PATH, 'vagas.parquet')}")
    if not applicants_df.empty:
        applicants_df.to_parquet(os.path.join(
            PROCESSED_DATA_PATH, "applicants.parquet"), index=True)
        print(
            f"  Candidatos salvos em {os.path.join(PROCESSED_DATA_PATH, 'applicants.parquet')}")
    if not prospects_df.empty:
        prospects_df.to_parquet(os.path.join(
            PROCESSED_DATA_PATH, "prospects.parquet"), index=True)
        print(
            f"  Prospects salvos em {os.path.join(PROCESSED_DATA_PATH, 'prospects.parquet')}")

    print("--- 2. Carregando Modelo de Embedding e Gerando Embeddings ---")
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("  Modelo de embedding 'all-MiniLM-L6-v2' carregado.")
    except Exception as e:
        print(f"  ERRO ao carregar o modelo de embedding: {e}")
        return  # Encerrar se o modelo não carregar

    if not jobs_df.empty:
        print("  Gerando embeddings para vagas...")
        vaga_texts = jobs_df['processed_text'].tolist()
        vaga_texts = [str(text) if pd.notna(
            text) else "" for text in vaga_texts]
        vaga_embeddings_array = embedding_model.encode(
            vaga_texts, show_progress_bar=True, convert_to_numpy=True)
        with open(VAGA_EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump({'ids': jobs_df.index.tolist(),
                        'embeddings': vaga_embeddings_array}, f)
        print(
            f"  {len(vaga_embeddings_array)} embeddings de vagas gerados e salvos em {VAGA_EMBEDDINGS_FILE}.")

    if not applicants_df.empty:
        print("  Gerando embeddings para candidatos...")
        candid_texts = applicants_df['processed_text'].tolist()
        candid_texts = [str(text) if pd.notna(
            text) else "" for text in candid_texts]
        candid_embeddings_array = embedding_model.encode(
            candid_texts, show_progress_bar=True, convert_to_numpy=True)
        with open(CANDID_EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump({'ids': applicants_df.index.tolist(),
                        'embeddings': candid_embeddings_array}, f)
        print(f"  {len(candid_embeddings_array)} embeddings de candidatos gerados e salvos em {CANDID_EMBEDDINGS_FILE}.")

    if not prospects_df.empty:
        print("  Gerando embeddings para prospects...")
        prospect_texts = prospects_df['processed_text'].tolist()
        prospect_texts = [str(text) if pd.notna(
            text) else "" for text in prospect_texts]
        prospect_embeddings_array = embedding_model.encode(
            prospect_texts, show_progress_bar=True, convert_to_numpy=True)
        with open(PROSPECT_EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump({'ids': prospects_df.index.tolist(),
                        'embeddings': prospect_embeddings_array}, f)
        print(f"  {len(prospect_embeddings_array)} embeddings de prospects gerados e salvos em {PROSPECT_EMBEDDINGS_FILE}.")

    print("\n--- Geração de dados pré-processados e embeddings CONCLUÍDA! ---")


if __name__ == "__main__":
    run_preprocessing_and_embedding_generation()

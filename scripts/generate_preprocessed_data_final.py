
import os
import pathlib
import re
import pandas as pd
import json
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import unicodedata
pd.set_option('display.max_columns', None)


print('Definindo as funcoes que serão utilizadas')


def carregar_json_com_dict(path):
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    return pd.DataFrame.from_dict(raw, orient='index')


def limpar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = unicodedata.normalize('NFKD', texto).encode(
        'ASCII', 'ignore').decode('utf-8', 'ignore')
    texto = texto.lower()
    texto = re.sub(r'[\n\r\t]+', ' ', texto)
    # texto = re.sub(r"[^a-zA-Z\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


def processing_applicants(embedding_model,
                          carregar_json_com_dict,
                          limpar_texto,
                          BASE_DATA_PATH,
                          PROCESSED_DATA_PATH,
                          CANDID_EMBEDDINGS_FILE):

    print('Processamento de applicants iniciado')

    df_applicants = carregar_json_com_dict(
        f"{BASE_DATA_PATH}/applicants.json")

    applicants_expand = df_applicants.apply(lambda row: {
        **row.get('infos_basicas', {}),
        **row.get('informacoes_pessoais', {}),
        **row.get('informacoes_profissionais', {}),
        **row.get('formacao_e_idiomas', {}),
        **row.get('cargo_atual', {}),
        'cv_pt': row.get('cv_pt', '')
    }, axis=1)
    df_applicants = pd.DataFrame(applicants_expand.tolist()).fillna('')

    df_applicants.insert(
        0, "id_candidato",
        df_applicants['codigo_profissional'],
    )

    print(f'Shape de applicants : {df_applicants.shape}')

    df_applicants.describe(include='all')

    cols_to_drop = [
        'telefone_recado', 'telefone', 'telefone_celular', 'data_criacao',
        'inserido_por', 'data_atualizacao', 'codigo_profissional',
        'data_aceite', 'cpf', 'fonte_indicacao', 'email_secundario',
        'data_nascimento', 'sexo', 'estado_civil', 'pcd', 'endereco',
        'skype', 'facebook', 'remuneracao', 'download_cv', 'outro_curso',
        'id_ibrati', 'email_corporativo', 'data_admissao', 'email', 'local',
        'data_ultima_promocao', 'nome_superior_imediato',
        'email_superior_imediato', 'inserido_por'
    ]

    df_applicants.drop(columns=cols_to_drop, axis=1, inplace=True)
    print(df_applicants.shape)

    print(f'Apenas candidatos únicos?'
          f'{df_applicants["id_candidato"].nunique() == df_applicants.shape[0]}')

    print('Tratando os textos para geração das embeddings posteriormente')

    for coluna in df_applicants.columns:
        print(f'Limpeza da coluna {coluna}')
        df_applicants[coluna] = df_applicants[coluna].apply(
            limpar_texto)

    print('Gerando texto único processado.')

    df_applicants.loc[slice(None), 'processed_text'] = df_applicants.apply(
        lambda x: ' '.join(filter(None, [*x])).strip(), axis=1)

    print('Exportando arquivo gerado em applicants inicialmente em parquet.')

    df_applicants.to_parquet(os.path.join(
        PROCESSED_DATA_PATH, 'applicants.parquet'), index=True)

    print('Gerando embeddings em df_applicants (candidatos)')

    candid_texts = df_applicants['processed_text'].tolist()

    candid_texts = [str(text)
                    if pd.notna(text)
                    else ""
                    for text in candid_texts
                    ]

    candid_embeddings_array = embedding_model.encode(
        candid_texts, show_progress_bar=True, convert_to_numpy=True)

    print('Exportando o arquivo de candidatos embeddado em pickle.')
    with open(CANDID_EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump({'ids': df_applicants.index.tolist(),
                     'embeddings': candid_embeddings_array}, f)

    print('Processamento de applicants concluído')


def processing_vagas(embedding_model, carregar_json_com_dict, limpar_texto, BASE_DATA_PATH, PROCESSED_DATA_PATH, VAGA_EMBEDDINGS_FILE):
    print('Iniciado processsamento de vagas')

    df_vagas = carregar_json_com_dict(
        f"{BASE_DATA_PATH}/vagas.json")

    vagas_expand = df_vagas.apply(lambda row: {
        **row.get('informacoes_basicas', {}),
        **row.get('perfil_vaga', {}),
        **row.get('beneficios', {}),
        'id_vaga': row.name
    }, axis=1)

    df_vagas = pd.DataFrame(vagas_expand.tolist()).fillna('')
    df_vagas["id_vaga"] = df_vagas["id_vaga"].astype(str)
    print(f'Shape de vagas : {df_vagas.shape}')

    cols_to_drop = [
        'solicitante_cliente', 'cliente', 'requisitante', 'analista_responsavel',
        'superior_imediato', 'origem_vaga', 'telefone', 'pais', 'local_trabalho',
        'nome_substituto'
    ]
    df_vagas.drop(columns=cols_to_drop, axis=1, inplace=True)

    print(f'Shape de vagas : {df_vagas.shape}')

    for coluna in df_vagas.columns:
        print(f'Limpeza da coluna {coluna}')
        df_vagas[coluna] = df_vagas[coluna].apply(
            limpar_texto)

    df_vagas['processed_text'] = df_vagas.apply(
        lambda x: ' '.join(filter(None, [*x])).strip(), axis=1)

    df_vagas.to_parquet(os.path.join(
        PROCESSED_DATA_PATH, "vagas.parquet"), index=True)

    print('Gerando embedding para vagas')

    vaga_texts = df_vagas['processed_text'].tolist()
    vaga_texts = [str(text) if pd.notna(
        text) else "" for text in vaga_texts]
    vaga_embeddings_array = embedding_model.encode(
        vaga_texts, show_progress_bar=True, convert_to_numpy=True)

    print('Exportando o arquivo de vagas embeddado em pickle.')

    with open(VAGA_EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump({'ids': df_vagas.index.tolist(),
                     'embeddings': vaga_embeddings_array}, f)


def processing_prospects(embedding_model, carregar_json_com_dict, limpar_texto, BASE_DATA_PATH, PROCESSED_DATA_PATH, PROSPECT_EMBEDDINGS_FILE):
    df_prospects = carregar_json_com_dict(
        f"{BASE_DATA_PATH}/prospects.json")

    df_prospects = df_prospects.explode('prospects')

    df_prospects['prospects'] = df_prospects['prospects'].fillna('')

    df_prospects['allblankorna'] = df_prospects.apply(
        lambda x: sum(x.isna()) + sum(x == ''), axis=1)

    df_prospects = df_prospects.loc[
        df_prospects.loc[
            slice(None), 'allblankorna'
        ] != 3, slice(None)
    ]
    df_prospects.insert(0, 'id_vaga_associada', df_prospects.index,
                        allow_duplicates=True)
    df_prospects['id_vaga_associada'] = df_prospects['id_vaga_associada'].astype(
        str)
    df_prospects.drop('allblankorna', axis=1, inplace=True)

    prospects_expand = df_prospects.apply(lambda row: {
        'id_vaga_associada': row.get('id_vaga_associada', ''),
        **row.get('prospects', {}),
        'titulo': row.get('titulo', ''),
        'modalidade': row.get('modalidade', '')
    }, axis=1)

    df_prospects = pd.DataFrame(prospects_expand.tolist()).fillna('')

    for coluna in df_prospects.columns:
        print(f'Limpeza da coluna {coluna}')
        df_prospects[coluna] = df_prospects[coluna].apply(
            limpar_texto)

    df_prospects.loc[slice(None), 'processed_text'] = df_prospects.apply(
        lambda x: ' '.join(filter(None, [*x])).strip(), axis=1)

    # df_prospects['id_prospect'] = df_prospects['codigo'].copy()
    df_prospects['id_prospect'] = df_prospects['codigo'].copy()
    df_prospects.drop(columns='codigo', inplace=True)
    df_prospects.set_index('id_prospect')

    print('Exportando arquivo gerado em prospect inicialmente em parquet.')

    df_prospects.to_parquet(os.path.join(
        PROCESSED_DATA_PATH, 'prospects.parquet'), index=True)

    print("Gerando embeddings para prospects...")
    prospect_texts = df_prospects['processed_text'].tolist()

    prospect_texts = [str(text) if pd.notna(
        text) else "" for text in prospect_texts]
    prospect_embeddings_array = embedding_model.encode(
        prospect_texts, show_progress_bar=True, convert_to_numpy=True)

    print('Exportando o arquivo de vagas embeddado em pickle.')

    with open(PROSPECT_EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump({'ids': df_prospects.index.tolist(),
                     'embeddings': prospect_embeddings_array}, f)


if __name__ == '__main__':

    print("Definicao dos caminhos que serão tratados e saídas geradas")

    BASE_DATA_PATH = 'data'

    PROCESSED_DATA_PATH = os.path.join(
        BASE_DATA_PATH, "processed_data")

    VAGA_EMBEDDINGS_FILE = os.path.join(
        PROCESSED_DATA_PATH, "vaga_embeddings.pkl")

    CANDID_EMBEDDINGS_FILE = os.path.join(
        PROCESSED_DATA_PATH, "candid_embeddings.pkl")

    PROSPECT_EMBEDDINGS_FILE = os.path.join(
        PROCESSED_DATA_PATH, "prospect_embeddings.pkl")

    print('Processamento da bases, feature engineering e exportação dos itens\
        que serão usados nos modelos')

    print('Setando o modelo que será usado para embeddings')

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    processing_applicants(
        embedding_model,
        carregar_json_com_dict,
        limpar_texto,
        BASE_DATA_PATH,
        PROCESSED_DATA_PATH,
        CANDID_EMBEDDINGS_FILE
    )

    processing_vagas(
        embedding_model,
        carregar_json_com_dict,
        limpar_texto,
        BASE_DATA_PATH,
        PROCESSED_DATA_PATH,
        VAGA_EMBEDDINGS_FILE
    )

    processing_prospects(
        embedding_model,
        carregar_json_com_dict,
        limpar_texto,
        BASE_DATA_PATH,
        PROCESSED_DATA_PATH,
        PROSPECT_EMBEDDINGS_FILE
    )

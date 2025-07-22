import streamlit as st
import pandas as pd
import numpy as np
from src.data_loader import load_processed_data
from src.nlp_matcher import (
    load_all_embeddings,
    find_top_matches,
    get_llm_explanation_for_match,
    get_single_embedding
)

st.set_page_config(layout='wide')

st.set_page_config(page_title='Projeto Datathon')
st.title('Matching de candidatos')

with st.spinner("Carregando embeddings pré-gerados..."):
    df_jobs, df_applicants, df_prospects = load_processed_data()


if df_jobs.empty or df_applicants.empty:
    st.error('Erro: Não foi possível carregar os dados essenciais')

st.success(f'Dados carregados com sucesso: ||'
           + f'Vagas:{len(df_jobs)} || Candidatos:{len(df_applicants)}||'
           + f'Prospects:{len(df_prospects)}')

with st.spinner("Carregando embeddings pré-gerados..."):
    embeddings_data = load_all_embeddings()

    # Extrai os arrays de embeddings e seus IDs correspondentes
    vaga_embeddings = embeddings_data['jobs']['embeddings']
    vaga_ids = embeddings_data['jobs']['ids']
    candid_embeddings = embeddings_data['applicants']['embeddings']
    candid_ids = embeddings_data['applicants']['ids']
    prospect_embeddings = embeddings_data['prospects']['embeddings']
    prospect_ids = embeddings_data['prospects']['ids']

if vaga_embeddings is None or candid_embeddings is None:
    st.error("Erro ao carregar embeddings. Verifique o módulo nlp_matcher e os logs.")
    st.stop()
st.success("Embeddings prontos para matching!")

st.header("Ferramenta de Matching")


job_display_names = [
    # f"{idx} - {jobs_df.loc[idx]['informacoes_basicas']['titulo_vaga']}" for idx in jobs_df.index]
    f"{valor['id_vaga']} - {valor['titulo_vaga']} " for _, valor in df_jobs.iterrows()
]
selected_job_display = st.selectbox("Selecione uma Vaga:", job_display_names)


selected_job_id = None
if selected_job_display:

    selected_job_id = selected_job_display.split(' - ')[0]
    selected_job = df_jobs.loc[
        df_jobs.loc[slice(None), 'id_vaga'] == selected_job_id,
        slice(None)]

    selected_job_id = selected_job.index

    # # Encontra o embedding da vaga selecionada usando o índice do ID na lista de IDs
    # # Assumimos que a ordem dos IDs em vaga_ids corresponde à ordem dos embeddings em vaga_embeddings
    try:
        job_embedding_idx = vaga_ids.index(selected_job_id)
        selected_job_embedding = vaga_embeddings[job_embedding_idx]
    except ValueError:
        st.error(
            f"Erro: Embedding para a vaga ID '{selected_job_id}' não encontrado. Pode ser um problema com os dados pré-gerados.")
        st.stop()

    st.markdown(
        f"**Vaga Selecionada:** {selected_job['titulo_vaga'].values}")

    st.markdown(f"**Descrição Processada da Vaga:**")
    # Mostra um pedaço da descrição processada
    st.write(selected_job['processed_text'][:500] + "...")

    match_type = st.radio(
        "Buscar Matches em:", ("Candidatos (applicants.json)", "Prospects (prospects.json)"))

    if st.button("Encontrar Melhores Matches"):
        if match_type == "Candidatos (applicants.json)":
            target_df = df_applicants
            target_embeddings_data = {
                'ids': candid_ids, 'embeddings': candid_embeddings}
            target_id_col = 'id_candidato'
            text_col = 'processed_text'

            # Função para obter o nome do candidato de forma segura

            def get_name(data): return data['infos_basicas']['nome'] if 'infos_basicas' in data and 'nome' in data[
                'infos_basicas'] else f"Candidato {data.get(target_id_col, 'N/A')}"
        else:  # Prospects
            target_df = df_prospects.copy()
            target_embeddings_data = {
                'ids': prospect_ids, 'embeddings': prospect_embeddings}

            # st.dataframe(target_df)

            target_id_col = 'id_prospect'
            text_col = 'processed_text'
            # # # Função para obter o nome do prospect de forma segura (ajuste conforme a real estrutura do seu prospects.json)

            def get_name(data): return data.get(
                'nome', f"Prospect {data.get(target_id_col, 'N/A')}")

        with st.spinner(f"Buscando {match_type} compatíveis..."):
            top_matches_df = find_top_matches(
                query_embedding=selected_job_embedding,
                target_embeddings_data=target_embeddings_data,
                top_n=5
            )

        if not top_matches_df.empty:
            st.write("---")  # Separador visual para os resultados
            for index, row in top_matches_df.iterrows():
                match_id = row['id']
                score = row['similarity_score']

                # Acessa os dados completos do candidato/prospect usando o ID
                match_data = target_df.loc[match_id] if match_id in target_df.index else target_df[
                    target_df[target_id_col] == match_id].iloc[0]

                entity_name = get_name(match_data)  # Obtém o nome formatado

                st.write(
                    f"**{match_type.replace(' (...', '')[:-1]}:** {entity_name} (ID: {match_id})")
                st.write(f"**Score de Similaridade:** {score:.4f}")

                # Mostra um pedaço do texto processado, substituido pelo texto tabular
                # st.write(
                #     f"**Texto Processado:** {match_data[text_col][:500]}...")

                st.write(match_data[:-1])

                # # --- LLM para Explicação do Match ---
                # # Cria uma chave única para o botão para evitar KeyErrors no Streamlit
                # button_key = f"explain_{match_type}_{match_id}"
                # if st.button(f"Explicar Match com {entity_name}", key=button_key):
                #     explanation = get_llm_explanation_for_match(
                #         job_text=selected_job['processed_text'],
                #         candidate_text=match_data[text_col],
                #         match_score=score
                #     )
                #     st.info(f"**Motivo da Seleção:** {explanation}")
                # st.write(f"---")  # Separador visual entre os matches
        else:
            st.info(
                f"Nenhum {match_type.replace(' (...', '')[:-1]} compatível encontrado para esta vaga.")

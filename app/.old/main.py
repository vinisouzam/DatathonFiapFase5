# app/main.py
import streamlit as st
from src.data_loader import load_all_data
from src.nlp_matcher import extract_text_from_job, compute_match
from src.chat_llm import ask_llm

st.set_page_config(layout="wide")
st.title("NLP Matching de Candidatos para Vagas")

# --- Carregamento de Dados ---
with st.status("Carregando dados...", expanded=True) as status:
    st.write("Buscando arquivos JSON...")
    jobs, applicants, _ = load_all_data()
    if jobs and applicants:
        status.update(label="Dados Carregados!",
                      state="complete", expanded=False)
        st.success(
            f"Dados carregados com sucesso! {len(jobs)} vagas e {len(applicants)} candidatos.")
    else:
        status.update(label="Falha no Carregamento de Dados!",
                      state="error", expanded=True)
        st.error("Erro: Não foi possível carregar os dados. Verifique os arquivos.")
        st.stop()

# --- Seção para a Interação com o LLM ---
user_prompt = st.text_input("Digite sua pergunta para o LLM:")

if st.button("Obter Resposta do LLM"):
    if user_prompt:
        with st.status("O LLM está processando sua pergunta...", expanded=True) as status_llm:
            status_llm.write(
                f"Iniciando chamada para LLM com prompt: '{user_prompt[:50]}...'")
            llm_response = ask_llm(prompt=user_prompt, max_tokens=200)
            status_llm.update(label="Resposta do LLM pronta!",
                              state="complete", expanded=False)

        st.subheader("Resposta do LLM:")
        st.write(llm_response)
    else:
        st.warning("Por favor, digite uma pergunta.")

# --- Exemplo para o match de vagas/candidatos ---
st.subheader("Simulação de Matching de Vagas")
if st.button("Executar Matching"):
    with st.status("Executando algoritmo de matching...", expanded=True) as status_match:
        status_match.write("Preparando dados para matching...")
        # ... (código de preparação) ...
        status_match.write("Calculando similaridades...")
        matched_results = compute_match(jobs, applicants)
        status_match.update(label="Matching concluído!",
                            state="complete", expanded=False)

    st.success("Matching concluído!")
    st.write("Resultados do Matching (Exemplo):")
    st.json(matched_results)


# from src.chat_llm import ask_llm
# from src.nlp_matcher import extract_text_from_job, compute_match
# from src.data_loader import load_all_data
# import streamlit as st


# # app/main.py

# st.title("NLP Matching de Candidatos para Vagas")

# jobs, applicants, _ = load_all_data()
# job_ids = list(jobs.keys())

# selected_job_id = st.selectbox("Selecione a vaga:", job_ids)
# job = jobs[selected_job_id]
# job_text = extract_text_from_job(job)
# st.markdown("**Descrição da vaga (processada):**")
# st.write(job_text)

# if st.button("🔎 Encontrar candidatos compatíveis"):
#     st.info("Processando embeddings e similaridade...")
#     ranked_matches = compute_match(job_text, applicants)

#     st.success("Top 5 candidatos mais compatíveis:")
#     for i, (cand_id, score) in enumerate(ranked_matches[:5]):
#         st.markdown(f"**{i+1}. Candidato {cand_id}** — Score: `{score:.2f}`")

#         if st.checkbox(f"Ver explicação do LLM - Candidato {cand_id}"):
#             cand_cv = applicants[cand_id].get("cv_pt", "")
#             prompt = f"""
# Você é um assistente de RH. Com base na descrição da vaga abaixo e no currículo do candidato, explique por que esse candidato é compatível com a vaga:

# Descrição da vaga:
# {job_text}

# Currículo do candidato:
# {cand_cv}

# Explique a compatibilidade de forma clara e objetiva.
# """
#             explanation = ask_llm(prompt)
#             st.write(explanation)

# # teste final

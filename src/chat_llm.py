from llama_cpp import Llama
import os
import streamlit as st

MODEL_PATH = os.path.join("models", "mistral-7b-openorca.Q4_0.gguf")

# Carregando o modelo para economizar processamento no momento de disponibilizar


@st.cache_resource(
    show_spinner="Carregando modelo de LLM... (primeira vez pode demorar)")
def load_llm_model():
    """carrega o modelo que será utilizado no LLM.
    """
    print(
        f"DEBUG_LLM: Carregando modelo LLM de: {MODEL_PATH} (Isso só deve acontecer uma vez por sessão/cache!)")
    try:
        llm_instance = Llama(
            model_path=MODEL_PATH,
            # tamanho máximo de tokens (será o interpretado).
            n_ctx=1024,
            # Número de threads da CPU para usar.
            n_threads=4,
            # Tamanho do batch para processamento de tokens. Ajuste se tiver problemas de memória.
            n_batch=512,
            # Bloqueia a memória para evitar swap, bom para performance, mas pode causar OOM se não houver RAM suficiente.
            use_mlock=True,
            chat_format="chatml"
            # n_gpu_layers=0 # Descomente e defina para o número de camadas que quer descarregar na GPU (Metal no M1/M2).
            # Se não tiver Metal configurado ou estiver tendo problemas, defina como 0.
        )
        print("DEBUG_LLM: Modelo LLM carregado com sucesso.")
        return llm_instance
    except Exception as e:
        print(f"DEBUG_LLM: ERRO ao carregar o modelo LLM: {e}")
        st.error(
            f"Erro ao carregar o modelo LLM: {e}. Verifique o caminho e os recursos do Docker Desktop.")
        # Impede que o aplicativo Streamlit continue se o LLM não puder ser carregado.
        st.stop()


# A instância do LLM será carregada e cacheada na primeira vez que o aplicativo rodar.
llm = load_llm_model()


def ask_llm(prompt: str, max_tokens=200):
    """
    Faz uma pergunta ao LLM no formato de chat e retorna a resposta.
    """
    with st.spinner("O LLM está pensando... Por favor, aguarde."):
        print(
            f"DEBUG_LLM: Gerando resposta com chat_format (max_tokens={max_tokens}).")
        try:
            messages = [
                {"role": "system", "content": "Você é um especialista em recrutamento que explica por que um candidato é compatível com uma vaga."},
                {"role": "user", "content": prompt}
            ]
            output = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            response_text = output["choices"][0]["message"]["content"].strip()
            print(
                f"DEBUG_LLM: Resposta do LLM gerada. (Primeiras 50 chars: {response_text[:50]})")
            return response_text
        except Exception as e:
            print(f"DEBUG_LLM: ERRO durante a inferência do LLM: {e}")
            st.error(f"Erro durante a inferência do LLM: {e}")
            return "Não foi possível gerar uma resposta. Tente novamente."


# %%%%%%%%%%%%%%%%%%%

# def ask_llm(prompt: str, max_tokens=200):
#     """
#     Faz uma pergunta ao LLM e retorna a resposta.
#     Adiciona um spinner na interface do Streamlit enquanto o LLM está processando.
#     """
#     with st.spinner("O LLM está pensando... Por favor, aguarde."):
#         print(
#             f"DEBUG_LLM: Gerando resposta para o prompt (max_tokens={max_tokens}).")
#         try:
#             # `stop` é útil para evitar que o LLM continue gerando texto indesejado
#             output = llm(prompt=prompt, max_tokens=max_tokens,
#                          stop=["\n", "###", "```"])
#             response_text = output["choices"][0]["text"].strip()
#             print(
#                 f"DEBUG_LLM: Resposta do LLM gerada. (Primeiras 50 chars: {response_text[:50]})")
#             return response_text
#         except Exception as e:
#             print(f"DEBUG_LLM: ERRO durante a inferência do LLM: {e}")
#             st.error(f"Erro durante a inferência do LLM: {e}")
#             return "Não foi possível gerar uma resposta. Tente novamente."


if __name__ == "__main__":
    # Este bloco é apenas para testar o módulo chat_llm independentemente do Streamlit.
    print("Testando chat_llm.py diretamente...")
    test_prompt = "O que é inteligência artificial?"
    response = ask_llm(test_prompt, max_tokens=50)
    print(f"Prompt de teste: {test_prompt}")
    print(f"Resposta: {response}")

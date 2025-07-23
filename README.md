# Match NLP App — FIAP Pós Datathon

Projeto de matching inteligente entre vagas e candidatos usando NLP com embeddings e modelo LLM local (GPT4All).

Criacao de dockerfile e docker-compose.yml pois estou com um Mac M1

Deploy em streamlit para podermos analisar os resultados.

- Caso os dados não estejam disponíveis na pasta, acessar ao link e realizar o download salvando em data
    link: https://drive.google.com/drive/folders/1f3jtTRyOK-PBvND3JTPTAxHpnSrH7rFR
    Observação : Implementado o download automático do hugging face para usar o streamlit, nesse link teremos 
    apenas os arquivos brutos.

- Rodar primeiro o script de pré - processamento que esta na pasta scripts, podem demorar a depender da capacidade da máquina, no meu caso demorou 7 minutos para vagas, 25 minutos para candidatos e 8 minutos para prospects;

joguei o app para fora das pastas para facilitar o entendimento do streamlit e permitir deploy

- Rodar o streamlit que o projeto já estará funcional;

# Link deployado do streamlit
https://datathonfiapfase5-n4vr6redzwnoasgyyqzlxv.streamlit.app/

## Devido limitaçao da ferramenta, ela funciona e depois quebra
No docker ele funcionou bem, retirei o LLM porque não dá para subir no modelo e precisava ainda ajustar 
todos os parametros, porque tenho um limite de memória no meu pc.

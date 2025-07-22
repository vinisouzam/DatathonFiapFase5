import os
import requests
from pathlib import Path

# def download_file(url: str, output_path: str):
#     """Faz o download do arquivo da URL se ainda não existir localmente."""
#     output_path = Path(output_path)
#     if not output_path.exists():
#         output_path.parent.mkdir(parents=True, exist_ok=True)
#         print(f"Baixando {url} para {output_path}...")
#         response = requests.get(url)
#         response.raise_for_status()
#         with open(output_path, "wb") as f:
#             f.write(response.content)
#     else:
#         print(f"Arquivo já existe em: {output_path}")


def download_file(url, output_path):
    if os.path.exists(output_path):
        return  # Arquivo já existe, não baixa de novo

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        raise Exception(f"Erro ao baixar {url}: {response.status_code}")

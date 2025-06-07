import os

import requests

url = "https://raw.githubusercontent.com/OSsari/CrisprBERT/main/data/Change_seq.csv"

local_path = "data/Change_seq.csv"

os.makedirs(os.path.dirname(local_path), exist_ok=True)

response = requests.get(url)
response.raise_for_status()

with open(local_path, "wb") as f:
    f.write(response.content)

print(f"File downloaded successfully to {local_path}")

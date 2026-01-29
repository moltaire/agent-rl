# scripts/download_lefebvre2017_data.py

import requests
from pathlib import Path
import zipfile

url = "https://ndownloader.figshare.com/files/6949427"
output_dir = Path("data/lefebvre2017")
output_dir.mkdir(parents=True, exist_ok=True)

print("Downloading...")
response = requests.get(url)
response.raise_for_status()

zip_path = output_dir / "temp.zip"
zip_path.write_bytes(response.content)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(output_dir)

zip_path.unlink()
print(f"Done: {output_dir}")
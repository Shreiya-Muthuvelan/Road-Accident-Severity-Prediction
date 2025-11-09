
from pathlib import Path

def download_dataset() -> str:
    import gdown
    sheet_id = "1_pS-4y7PfovLJoWlLAvVAizvk1iScFZ7"
    gid = "545325522"

    download_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}" 
    output_path = Path("data/raw/dataset.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file = str(output_path)

    gdown.download(url=download_url, output=output_file, quiet=False)
    return output_file



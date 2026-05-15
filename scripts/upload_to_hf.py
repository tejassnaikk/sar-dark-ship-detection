"""
One-shot script: upload best.pt and model card to HF Hub.

Run from project root:
    python scripts/upload_to_hf.py

Requires: huggingface_hub installed + `hf auth login` completed.
"""
from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO_ID      = "tejassnaikk/rsdd-yolo11n-obb-v1"
WEIGHTS_PATH = "models_local/best.pt"
CARD_PATH    = "docs/model_card.md"


def main() -> None:
    weights = Path(WEIGHTS_PATH)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights.resolve()}")
    if weights.stat().st_size < 1_000_000:
        raise ValueError(f"Weights look too small ({weights.stat().st_size} bytes) — check path.")

    api = HfApi()

    print(f"Creating repo {REPO_ID} (no-op if already exists) …")
    create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

    print(f"Uploading {WEIGHTS_PATH} …")
    api.upload_file(
        path_or_fileobj=str(weights),
        path_in_repo="best.pt",
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"  ✓ weights uploaded")

    card = Path(CARD_PATH)
    if card.exists():
        print(f"Uploading {CARD_PATH} as README.md …")
        api.upload_file(
            path_or_fileobj=str(card),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="model",
        )
        print(f"  ✓ model card uploaded")
    else:
        print(f"  ⚠ {CARD_PATH} not found — skipping model card upload")

    print(f"\nDone. View at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()

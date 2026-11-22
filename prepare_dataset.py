from pathlib import Path
import json

ROOT = Path(__file__).parent
with open(ROOT / "characters.json", encoding="utf-8") as f:
    characters = json.load(f)["characters"]

image_root = ROOT / "dataset" / "images"
audio_root = ROOT / "dataset" / "audio"
image_root.mkdir(parents=True, exist_ok=True)
audio_root.mkdir(parents=True, exist_ok=True)

for item in characters:
    (image_root / item["label"]).mkdir(exist_ok=True)

print("Created image folders for:")
for item in characters:
    print(f"  {item['label']} ({item['roman']})")
print("\nAdd handwritten JPG/PNG files to the matching folder.")
print("Add a clear Kannada voice recording to:")
for item in characters:
    print(f"  dataset/audio/{item['label']}.wav")

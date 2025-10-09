from pathlib import Path
import json
import pandas as pd
import os

_FILE_DIR = Path(__file__).resolve().parent.parent#obtain directory of this file
_PROJ_DIR = _FILE_DIR.parent#obtain main project directory
_DATA_DIR = _PROJ_DIR / "dataset"
_EXTRACT_DIR = _PROJ_DIR / "dataExtracted"

nameSet = set()

# Loop through all files in the directory
for filename in os.listdir(str(_DATA_DIR / "cablesGeojson")):
    if filename.endswith(".geojson"):
        filepath = os.path.join(str(_DATA_DIR / "cablesGeojson"), filename)

        # Load geojson file
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract cable name
        for feature in data["features"]:
            name = feature["properties"].get("Name", "Unknown")

            nameSet.add(name)

#write to csv file
df = pd.DataFrame(nameSet, columns=["name"])
df.to_csv(str(_EXTRACT_DIR / "all_cables.csv"), index=False)
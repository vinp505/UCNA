from pathlib import Path
import pandas as pd

_FILE_DIR = Path(__file__).resolve().parent.parent#obtain directory of this file
_PROJ_DIR = _FILE_DIR.parent#obtain main project directory
_DATA_DIR = _PROJ_DIR / "dataset"
_EXTRACT_DIR = _PROJ_DIR / "dataExtracted"

def extractCapacities():
    #load the html file:
    tables = pd.read_html(str(_DATA_DIR / "capacityTable.html"))#get the capacity table from the data directory
    df = tables[0]#get the first table
    df = df[df["Status"] == "Active"]#only get the active cables
    df = df[["Name", "Short Name", "Capacity(in Gbps)"]]#get only interesting columns
    df.columns = ["name", "shortName", "capacity"]#rename columns
    #write to the extracted directory
    df.to_csv(str(_EXTRACT_DIR / "cableCapacities.csv"), index=False)
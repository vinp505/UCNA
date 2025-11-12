from pathlib import Path
import numpy as np
import pandas as pd

_FILE_DIR = Path(__file__).resolve().parent.parent#obtain directory of this file
_PROJ_DIR = _FILE_DIR.parent#obtain main project directory
#_DATA_DIR = _PROJ_DIR / "dataset"
_EXTRACT_DIR = _PROJ_DIR / "dataExtracted"

def mergeCapacities():
    #load the cable capacities
    capacitiesDf = pd.read_csv(str(_EXTRACT_DIR / "cableCapacities.csv"))
    cablesDf = pd.read_csv(str(_EXTRACT_DIR / "all_cables.csv"))

    print(f"Number of unique cables edges: {cablesDf.shape[0]}")
    filteredCapacities = capacitiesDf[capacitiesDf["capacity"] != 0]
    print(f"Number of capacity entries: {filteredCapacities.shape[0]}")
    capacityNames = set(filteredCapacities["name"].unique())
    capacityShortNames = set(filteredCapacities["shortName"].unique())

    # now let's try to merge on name and short name:
    # First join on Name
    merged1 = cablesDf.merge(filteredCapacities[["name", "capacity"]], 
                        on="name", how="left")

    # Then join on Short Name (only fill NaN values from first merge)
    finalDf = merged1.merge(filteredCapacities[["shortName", "capacity"]], 
                            left_on="name", right_on="shortName", 
                            how="left", suffixes=("", "_short"))

    # If Capacity from Name match is missing, take Capacity from Short Name match
    finalDf["capacity"] = finalDf["capacity"].fillna(finalDf["capacity_short"])

    # Drop helper column
    finalDf = finalDf.drop(columns=["shortName", "capacity_short"])
    finalDf["capacity"] = finalDf["capacity"].fillna(0)

    finalDf.to_csv(str(_EXTRACT_DIR / "cablesWithCapacity.csv"), index=False)

    print(f"Capacity matched for {np.sum(finalDf['capacity'] != 0.0)} cables.")#we have the capacities for 109 out of the 245 cables
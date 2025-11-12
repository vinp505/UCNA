from pathlib import Path

#own imports
from extract_cables import extractCableNames
from extractLandpoints import landpoints_to_csv
from extractCapacities import extractCapacities
from mergeCapacities import mergeCapacities

_FILE_DIR = Path(__file__).resolve().parent.parent#obtain directory of this file
_PROJ_DIR = _FILE_DIR.parent#obtain main project directory
_DATA_DIR = _PROJ_DIR / "dataset"
_EXTRACT_DIR = _PROJ_DIR / "dataExtracted"

extractCableNames()#extract the cable names from the .geojson files and put them into a csv file (this only extracts the names)
#extract longitude, latitude, name & description (which is city name + country) for all the landpoints from the .geojson file
landpoints_to_csv(str(_DATA_DIR / "landpointsGeojson" / "Landing_Points.geojson"), str(_EXTRACT_DIR / "landpoints.csv"))
extractCapacities()#extract the capacities with cable name & short cable name from the html table
mergeCapacities()#merge the cable names to the capacities extracted earlier and put everything into a .csv file
#NOW:   run the buildGraph.ipynb to create the network file
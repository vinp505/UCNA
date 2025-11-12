from pathlib import Path
import json
import pandas as pd

_FILE_DIR = Path(__file__).resolve().parent.parent#obtain directory of this file
_PROJ_DIR = _FILE_DIR.parent#obtain main project directory
_DATA_DIR = _PROJ_DIR / "dataset"
_EXTRACT_DIR = _PROJ_DIR / "dataExtracted"

def normCoord(long, lat):
    """Converts floating point coordinates to integers for easier comparisons!"""
    return (int(round(long * 1e6)), int(round(lat * 1e6)))

def landpoints_to_csv(geojson_file_path, csv_file_path):
    """
    Convert a GeoJSON file to CSV with coordinates, name and description
    
    Parameters:
    geojson_file_path (str): Path to input GeoJSON file
    csv_file_path (str): Path to output CSV file
    """
    
    # Read the GeoJSON file
    with open(geojson_file_path, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
    
    # Extract features
    features = geojson_data.get('features', [])
    
    # Prepare data for DataFrame
    data = []
    for feature in features:
        properties = feature.get('properties', {})
        geometry = feature.get('geometry', {})
        
        # Extract coordinates (assuming Point geometry)
        coordinates = geometry.get('coordinates', [])
        longitude = coordinates[0] if len(coordinates) > 0 else None
        latitude = coordinates[1] if len(coordinates) > 1 else None
        longitude, latitude = normCoord(longitude, latitude)
        
        # Extract name and description
        name = properties.get('Name', '')
        description = properties.get('description', '')
        
        data.append({
            'longitude': longitude,
            'latitude': latitude,
            'name': name,
            'description': description
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df = df[df["name"].notna()].copy().reset_index()#drop nas
    df.drop("index", axis=1, inplace=True)#html table brings some weird index column
    
    # Save to CSV
    df.to_csv(csv_file_path, index=False, encoding='utf-8')
    
    print(f"Successfully converted {len(data)} entries to {csv_file_path}")
    return df
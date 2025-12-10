#visualize the graph with folium
import folium
import os
import argparse
import os
from pathlib import Path
import networkx as nx

_FILE_DIR = Path(__file__).resolve().parent.parent#obtain directory of this file
_PROJ_DIR = _FILE_DIR.parent#obtain main project directory
_DATA_DIR = _PROJ_DIR / "dataset"
_EXTRACT_DIR = _PROJ_DIR / "dataExtracted"
_VISUAL_DIR = _PROJ_DIR / "visualizations"

def plot_graph_folium(G, output_file, cablesDir = None):
    # Extract coordinates from node attributes
    coords = [(d["lon"], d["lat"]) for _, d in G.nodes(data=True)]
    lons, lats = zip(*coords)
    
    # Center map on average position
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=3, tiles="CartoDB positron")

    #overlay the "real network"
    if cablesDir is not None:
        # Loop through all files in the directory
        for filename in os.listdir(cablesDir):
            if filename.endswith(".geojson"):
                filepath = os.path.join(cablesDir, filename)
                folium.GeoJson(filepath, color= "#9A8D8D").add_to(m)

    # Add edges
    for u, v, data in G.edges(data=True):
        lon1, lat1 = G.nodes[u]["lon"], G.nodes[u]["lat"]
        lon2, lat2 = G.nodes[v]["lon"], G.nodes[v]["lat"]
        html = f"""
        <div style="white-space: nowrap;">
        Edge {G.nodes[u]['city']} ←→ {G.nodes[v]['city']}<br>{data['capacity']} Gbps
        </div>
        """
        popup = folium.Popup(html, max_width=500)
        folium.PolyLine(
            locations=[(lat1, lon1), (lat2, lon2)],
            color= "#738CB6",
            highlight_color = "yellow",
            weight=2,
            popup=popup,
            opacity=0.6
        ).add_to(m)

    # Add nodes
    for n, d in G.nodes(data=True):
        
        if d['meta']:
            lon, lat = d["lon"], d["lat"]
            country = d['country']
            lon_str, lat_str = f"lon: {lon:.3f}", f"lat: {lat:.3f}"
            html = f"""
            <div style="white-space: nowrap;">
            {n}<br>{country}<br>{lon_str}<br>{lat_str}
            </div>
            """
            popup = folium.Popup(html, max_width=500)
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                popup=popup,
                color="#2E4057",
                highlight_color = "yellow",
                fill=True,
                fill_color="#2E4057",
                fill_opacity=1
            ).add_to(m)

        else:
            lon, lat = d["lon"], d["lat"]
            city, country = d['city'], d['country']
            lon_str, lat_str = f"lon: {lon:.3f}", f"lat: {lat:.3f}"
            html = f"""
            <div style="white-space: nowrap;">
            Node {n}<br>{city}<br>{country}<br>{lon_str}<br>{lat_str}
            </div>
            """
            popup = folium.Popup(html, max_width=500)
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                popup=popup,
                color="#F4A261",
                highlight_color = "yellow",
                fill=True,
                fill_color="#F4A261",
                fill_opacity=1
            ).add_to(m)

    # Save map as HTML
    m.save(output_file)
    print(f"Map saved to {output_file}")

def main():
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Open and process a file provided by CLI.")

    # Add a positional argument for the filename
    parser.add_argument("filename", type=str, help="The path to the file you want to open")

    # Parse the arguments
    args = parser.parse_args()

    # Check validity (optional but good practice)
    if not os.path.exists(str(_EXTRACT_DIR / str(args.filename))):
        print(f"Error: The file '{args.filename}' does not exist.")
        raise ValueError("Path doesn't exist.")

    #open file
    print(f"Opening {str(_EXTRACT_DIR / str(args.filename))}...")
    G = nx.read_graphml(str(_EXTRACT_DIR / str(args.filename)))

    #save in the visualizations directory
    plot_graph_folium(G, output_file=str(_VISUAL_DIR / (str(args.filename).split(sep='.')[0] + ".html")))

if __name__ == "__main__":
    main()
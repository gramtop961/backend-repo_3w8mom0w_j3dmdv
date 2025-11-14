import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from io import BytesIO
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}

@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}

@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    
    try:
        # Try to import database module
        from database import db
        
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            
            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
            
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    
    # Check environment variables
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    
    return response


# -----------------------
# Download helpers
# -----------------------

def build_notebook_json():
    """Create the ESDA Assignment 1 Jupyter notebook as a dict (nbformat 4)."""
    md_intro = """
# Exploratory Spatial Data Analysis – Assignment 1

Project Title: Spatial Relationship between Urban Populations and Earthquake Hazards

This notebook investigates how global population centers relate spatially to earthquake hazards. It covers:

1. Data exploration and preparation
2. Exploring population characteristics
3. Identifying unusual or extreme locations
4. Exploring earthquake distribution
5. Measuring the geographic distribution of events
6. Relating earthquakes to populated areas

Run the cells in order. The notebook will download open datasets at runtime.
"""

    md_data = """
## 1) Data Exploration and Preparation

Datasets used:
- Natural Earth Populated Places (1:10m): https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places_simple.zip
- Natural Earth Admin 0 Countries (1:10m): https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip
- USGS Earthquakes (last 30 days): https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.geojson

We ensure consistent CRS (WGS84, EPSG:4326), inspect schema, handle missing/duplicates, and quick visualization sanity checks.
"""

    code_setup = """
# Install required libraries if missing (safe to re-run)
import sys, subprocess

def pip_install(pkg):
    subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '--quiet'], check=False)

for pkg in [
    'pandas', 'geopandas', 'shapely', 'pyproj', 'matplotlib', 'seaborn',
    'contextily', 'mapclassify', 'folium', 'requests'
]:
    pip_install(pkg)

import os
import json
import math
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import folium
from folium.plugins import MarkerCluster, HeatMap
from zipfile import ZipFile
from io import BytesIO

plt.rcParams['figure.figsize'] = (10,6)
plt.rcParams['axes.grid'] = True
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
"""

    code_download = """
# Helper to read Natural Earth zipped shapefiles directly from URL
NE_PLACES = 'https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places_simple.zip'
NE_COUNTRIES = 'https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip'
USGS_EQ = 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.geojson'

def read_ne_zip(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    z = ZipFile(BytesIO(r.content))
    # find the .shp inside
    shp_name = [n for n in z.namelist() if n.endswith('.shp')][0]
    return gpd.read_file(f'/vsizip/{BytesIO(r.content).getbuffer().obj.name}')
"""

    # Using fiona vsizip path directly from BytesIO is tricky; use temp extract instead
    code_download2 = """
import tempfile

def read_ne_zip(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with tempfile.TemporaryDirectory() as td:
        z = ZipFile(BytesIO(r.content))
        z.extractall(td)
        shp = [os.path.join(td, f) for f in os.listdir(td) if f.endswith('.shp')][0]
        gdf = gpd.read_file(shp)
    return gdf

# Load datasets
places = read_ne_zip(NE_PLACES)
countries = read_ne_zip(NE_COUNTRIES)
eq = gpd.read_file(USGS_EQ)

# Ensure CRS alignment
places = places.to_crs('EPSG:4326')
countries = countries.to_crs('EPSG:4326')
eq = eq.to_crs('EPSG:4326')

print('Rows -> places:', len(places), 'countries:', len(countries), 'earthquakes:', len(eq))
print('CRS ->', places.crs, countries.crs, eq.crs)

# Basic cleaning: drop duplicates by name/coords for places
places['coord_key'] = places.geometry.apply(lambda g: (round(g.y,4), round(g.x,4)))
places = places.drop_duplicates(subset=['name', 'coord_key']).drop(columns=['coord_key'])

# Keep essential columns
place_cols = ['name', 'adm0name', 'pop_max', 'pop_min', 'featurecla', 'geometry']
places = places[place_cols]

# Earthquake essential columns
keep_eq = ['mag','place','time','url','geometry','type']
eq = eq[keep_eq]

# Drop rows with missing geometry
places = places[places.geometry.notna()].reset_index(drop=True)
eq = eq[eq.geometry.notna()].reset_index(drop=True)

# Quick sanity map
ax = countries.plot(facecolor='none', edgecolor='lightgray', linewidth=0.5)
places.sample(min(3000, len(places))).plot(ax=ax, color='dodgerblue', markersize=2, alpha=0.6)
plt.title('Natural Earth Populated Places (sample) over Countries')
plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/sanity_places.png', dpi=150)
plt.show()

ax = countries.plot(facecolor='none', edgecolor='lightgray', linewidth=0.5)
eq.plot(ax=ax, color='crimson', markersize=3, alpha=0.6)
plt.title('USGS Earthquakes (last 30 days)')
plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/sanity_eq.png', dpi=150)
plt.show()
"""

    md_pop = """
## 2) Exploring Population Characteristics
- Examine population distributions (pop_max)
- Transformations and visualization (log-scale)
- Identify unusual high/low values
"""

    code_pop = """
# Basic stats
pop = places['pop_max'].dropna()
print(pop.describe(percentiles=[.75,.9,.95,.99]))

fig, ax = plt.subplots()
sns.histplot(pop, bins=50, kde=True, ax=ax)
ax.set_title('City Population (pop_max) - linear scale')
plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/pop_hist_linear.png', dpi=150)
plt.show()

fig, ax = plt.subplots()
sns.histplot(np.log10(pop[pop>0]), bins=50, kde=True, ax=ax)
ax.set_title('City Population (log10 pop_max)')
plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/pop_hist_log.png', dpi=150)
plt.show()

# Z-score based outliers
z = (pop - pop.mean())/pop.std(ddof=1)
places['z_pop'] = ((places['pop_max'] - pop.mean())/pop.std(ddof=1)).fillna(0)

iqr = pop.quantile(0.75) - pop.quantile(0.25)
upper = pop.quantile(0.75) + 1.5*iqr
lower = max(pop.quantile(0.25) - 1.5*iqr, 0)

high_cities = places[places['pop_max'] >= upper].sort_values('pop_max', ascending=False).head(20)
low_cities = places[(places['pop_max'] <= lower)].head(20)
print('High outliers (IQR):', len(high_cities))
print('Low outliers (IQR):', len(low_cities))

# Map top 50 largest cities
top50 = places.sort_values('pop_max', ascending=False).head(50)
ax = countries.plot(facecolor='none', edgecolor='lightgray', linewidth=0.5)
top50.plot(ax=ax, color='gold', markersize=20, alpha=0.8, edgecolor='black')
plt.title('Top 50 Largest Cities (by pop_max)')
plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/top50_cities.png', dpi=150)
plt.show()
"""

    md_unusual = """
## 3) Identifying Unusual or Extreme Locations
We consider two notions:
- Attribute outliers: unusually large/small populations (z-score, IQR)
- Spatial isolation: cities far from their nearest neighbor
"""

    code_unusual = """
from sklearn.neighbors import BallTree
# Compute nearest neighbor distance among cities (in degrees -> approx km using haversine)
# Prepare radians for haversine
coords = np.radians(np.column_stack([places.geometry.y.values, places.geometry.x.values]))
if len(coords) > 1:
    tree = BallTree(coords, metric='haversine')
    dist, idx = tree.query(coords, k=2)  # nearest other point
    # convert to km (earth radius ~6371 km)
    nn_km = dist[:,1] * 6371
    places['nn_km'] = nn_km
else:
    places['nn_km'] = np.nan

iso_threshold = places['nn_km'].quantile(0.95)
isolated = places[places['nn_km'] >= iso_threshold]
print('Isolated cities (95th percentile by NN distance):', len(isolated))

ax = countries.plot(facecolor='none', edgecolor='lightgray', linewidth=0.5)
places.plot(ax=ax, color='lightgray', markersize=1, alpha=0.4)
isolated.plot(ax=ax, color='orange', markersize=10, alpha=0.9)
plt.title('Geographically Isolated Cities (top 5% by NN distance)')
plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/isolated_cities.png', dpi=150)
plt.show()
"""

    md_eq = """
## 4) Exploring Earthquake Distribution
Describe magnitudes, visualize global distribution, and compare across regions.
"""

    code_eq = """
# Earthquake magnitude summary
mag = eq['mag'].dropna()
print(mag.describe(percentiles=[.75,.9,.95,.99]))

fig, ax = plt.subplots()
sns.histplot(mag, bins=40, kde=True, ax=ax)
ax.set_title('Earthquake Magnitudes (last 30 days)')
plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/eq_mag_hist.png', dpi=150)
plt.show()

# Heatmap with Folium
m = folium.Map(location=[20,0], zoom_start=2, tiles='cartodbpositron')
HeatMap([[g.y, g.x] for g in eq.geometry]).add_to(m)
m.save(f'{OUTPUT_DIR}/eq_heatmap.html')

# Points with cluster
m2 = folium.Map(location=[20,0], zoom_start=2, tiles='cartodbpositron')
mc = MarkerCluster().add_to(m2)
for i, row in eq.iterrows():
    folium.CircleMarker(location=[row.geometry.y, row.geometry.x], radius=max(1, (row['mag'] or 0)),
                        color='crimson', fill=True, fill_opacity=0.6,
                        popup=f"Mag {row['mag']} - {row['place']}").add_to(mc)
    
m2.save(f'{OUTPUT_DIR}/eq_markers.html')
"""

    md_geo_dist = """
## 5) Measuring the Geographic Distribution of Events
Compute mean centers and standard distance for both cities and earthquakes.
Compare orientation using covariance ellipses.
"""

    code_geo_dist = """
from numpy.linalg import eig


def mean_center(gdf):
    xs = gdf.geometry.x.values
    ys = gdf.geometry.y.values
    return np.mean(xs), np.mean(ys)


def std_distance(gdf):
    xs = gdf.geometry.x.values
    ys = gdf.geometry.y.values
    mx, my = np.mean(xs), np.mean(ys)
    sd = np.sqrt(((xs-mx)**2 + (ys-my)**2).mean())
    return sd


def covariance_ellipse(gdf, n_std=1.0):
    xs = gdf.geometry.x.values
    ys = gdf.geometry.y.values
    cov = np.cov(np.vstack((xs, ys)))
    vals, vecs = eig(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    width, height = 2*n_std*np.sqrt(vals)
    angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    return width, height, angle

mc_places = mean_center(places)
mc_eq = mean_center(eq)
sd_places = std_distance(places)
sd_eq = std_distance(eq)
ell_places = covariance_ellipse(places, n_std=1.0)
ell_eq = covariance_ellipse(eq, n_std=1.0)

print('Mean center (cities):', mc_places, 'StdDist:', sd_places)
print('Mean center (eq):', mc_eq, 'StdDist:', sd_eq)
print('Ellipse (cities):', ell_places)
print('Ellipse (eq):', ell_eq)

# Plot centers and circles (approx visualization)
ax = countries.plot(facecolor='none', edgecolor='lightgray', linewidth=0.5)
places.sample(min(3000, len(places))).plot(ax=ax, color='steelblue', markersize=2, alpha=0.5)
eq.sample(min(3000, len(eq))).plot(ax=ax, color='crimson', markersize=2, alpha=0.5)
ax.scatter([mc_places[0]], [mc_places[1]], s=100, c='blue', marker='x', label='Cities Center')
ax.scatter([mc_eq[0]], [mc_eq[1]], s=100, c='red', marker='x', label='EQ Center')
plt.legend()
plt.title('Mean Centers of Cities and Earthquakes')
plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/mean_centers.png', dpi=150)
plt.show()
"""

    md_relate = """
## 6) Relating Earthquakes to Populated Areas
Compute nearest earthquake distance per city and summarize by country/region.
"""

    code_relate = """
# Build BallTree for earthquake points (haversine)
coords_eq = np.radians(np.column_stack([eq.geometry.y.values, eq.geometry.x.values]))
coords_pl = np.radians(np.column_stack([places.geometry.y.values, places.geometry.x.values]))

if len(coords_eq) > 0 and len(coords_pl) > 0:
    tree_eq = BallTree(coords_eq, metric='haversine')
    dist, idx = tree_eq.query(coords_pl, k=1)
    near_km = dist[:,0] * 6371
    places['nearest_eq_km'] = near_km
else:
    places['nearest_eq_km'] = np.nan

# Join country names and summarize exposure
cities = places[['name','adm0name','pop_max','nearest_eq_km','geometry']].copy()
exposure_summary = cities.groupby('adm0name').agg(
    cities=('name','count'),
    pop_sum=('pop_max','sum'),
    near_km_med=('nearest_eq_km','median'),
    near_km_p10=('nearest_eq_km', lambda s: s.quantile(0.10)),
    near_km_p90=('nearest_eq_km', lambda s: s.quantile(0.90)),
).reset_index().sort_values('near_km_med')

print(exposure_summary.head(15))

# Map cities colored by proximity
ax = countries.plot(facecolor='none', edgecolor='lightgray', linewidth=0.5)
sc = cities.plot(ax=ax, column='nearest_eq_km', cmap='viridis_r', markersize=5, legend=True)
plt.title('City Proximity to Nearest Earthquake (km)')
plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/city_proximity.png', dpi=150)
plt.show()

# Save outputs
cities.to_file(f'{OUTPUT_DIR}/cities_with_nearest_eq.geojson', driver='GeoJSON')
exposure_summary.to_csv(f'{OUTPUT_DIR}/exposure_by_country.csv', index=False)
"""

    md_final = """
## Final Analysis and Reporting
- The figures saved in the outputs/ folder illustrate population distributions, unusual cities, earthquake patterns, and proximity maps.
- Use the printed tables to identify countries with higher exposure (small median nearest_eq_km).

To export this notebook as a PDF with outputs, you can use:

- Option A (HTML): File -> Save and Export Notebook As -> HTML, then print to PDF.
- Option B (nbconvert): `pip install nbconvert` then run in a terminal:
  `jupyter nbconvert --to pdf --execute --output ESDA_Assignment1.pdf this_notebook.ipynb`

Include in your written report:
- Where population centers are most concentrated
- Unusually large or isolated cities
- Where people are most exposed to seismic hazards
- How global urban distribution aligns with earthquake zones
"""

    cells = []
    def md(s):
        return {"cell_type":"markdown","metadata":{},"source":s.strip().split('\n')}
    def code(s):
        return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip().split('\n')}

    for c in [md(md_intro), md(md_data), code(code_setup), code(code_download), code(code_download2), md(md_pop), code(code_pop), md(md_unusual), code(code_unusual), md(md_eq), code(code_eq), md(md_geo_dist), code(code_geo_dist), md(md_relate), code(code_relate), md(md_final)]:
        cells.append(c)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    return nb


@app.get('/api/download/notebook')
def download_notebook():
    nb = build_notebook_json()
    data = json.dumps(nb, indent=2).encode('utf-8')
    buf = BytesIO(data)
    headers = {
        'Content-Disposition': 'attachment; filename="ESDA_Assignment1.ipynb"'
    }
    return StreamingResponse(buf, media_type='application/x-ipynb+json', headers=headers)


@app.get('/api/download/report')
def download_report_template():
    # Generate a simple PDF report template with placeholders
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch
    except Exception:
        # If reportlab is not available, return a plain text fallback
        content = (
            "ESDA Assignment 1 Report Template\n\n"
            "Please install reportlab to get a PDF template.\n"
            "Sections:\n"
            "1. Introduction\n2. Data & Methods\n3. Results (maps, charts)\n4. Discussion\n5. Conclusion\n"
        )
        return PlainTextResponse(content, media_type='text/plain', headers={'Content-Disposition': 'attachment; filename="ESDA_Report_Template.txt"'})

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    width, height = LETTER

    def h(text, y):
        c.setFont("Helvetica-Bold", 14)
        c.drawString(1*inch, y, text)
        c.setFont("Helvetica", 11)
        return y-18

    y = height - 1*inch
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1*inch, y, "Exploratory Spatial Data Analysis – Assignment 1")
    y -= 24
    c.setFont("Helvetica", 11)
    c.drawString(1*inch, y, "Spatial Relationship between Urban Populations and Earthquake Hazards")

    y -= 36
    y = h("1. Introduction", y)
    c.drawString(1*inch, y, "Objective and overview of the analysis.")
    y -= 24
    y = h("2. Data & Preparation", y)
    c.drawString(1*inch, y, "Datasets, cleaning steps, CRS alignment.")
    y -= 24
    y = h("3. Population Characteristics", y)
    c.drawString(1*inch, y, "Distribution, transformations, unusual cities.")
    y -= 24
    y = h("4. Earthquake Distribution", y)
    c.drawString(1*inch, y, "Magnitude stats, spatial patterns, dense/sparse regions.")
    y -= 24
    y = h("5. Geographic Distribution Metrics", y)
    c.drawString(1*inch, y, "Mean centers, spread, ellipses, overlap.")
    y -= 24
    y = h("6. Earthquakes vs Populated Areas", y)
    c.drawString(1*inch, y, "Nearest distances, exposed regions, country comparisons.")
    y -= 24
    y = h("7. Results & Discussion", y)
    c.drawString(1*inch, y, "Key maps, charts, and interpretation.")
    y -= 24
    y = h("8. Conclusion", y)
    c.drawString(1*inch, y, "Summary of findings and limitations.")

    c.showPage()
    c.save()
    buf.seek(0)

    headers = {'Content-Disposition': 'attachment; filename="ESDA_Report_Template.pdf"'}
    return StreamingResponse(buf, media_type='application/pdf', headers=headers)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

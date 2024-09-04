import ee
import geemap
import json
import rasterio
from rasterio.merge import merge
import glob
from concurrent.futures import ThreadPoolExecutor

# Connect GEE API
GEE_CREDENTIALS_FILE = "./data/gee/geospatial_api_key.json"
credentials = ee.ServiceAccountCredentials(None, GEE_CREDENTIALS_FILE)
ee.Initialize(credentials)


geojson_file = "data/gee/shape/plot01.geojson"
with open(geojson_file) as f:
    geojson_data = json.load(f)

polygon = ee.Geometry.Polygon(geojson_data['features'][0]['geometry']['coordinates'])

def create_grid(polygon, grid_size):
    bounds = polygon.bounds().getInfo()['coordinates'][0]
    min_lon, min_lat = bounds[0]
    max_lon, max_lat = bounds[2]

    lons = [min_lon + i * grid_size for i in range(int((max_lon - min_lon) / grid_size) + 1)]
    lats = [min_lat + i * grid_size for i in range(int((max_lat - min_lat) / grid_size) + 1)]

    grids = []
    for i in range(len(lons) - 1):
        for j in range(len(lats) - 1):
            grid = ee.Geometry.Polygon([[
                [lons[i], lats[j]],
                [lons[i+1], lats[j]],
                [lons[i+1], lats[j+1]],
                [lons[i], lats[j+1]],
                [lons[i], lats[j]]
            ]])
            grids.append(grid)
    
    return ee.FeatureCollection(grids)

grid_size = 0.1
grid = create_grid(polygon, grid_size)

collection = (
    ee.ImageCollection('COPERNICUS/S2')
    .filterBounds(polygon)
    .filterDate('2024-08-01', '2024-08-31')
    .sort('CLOUDY_PIXEL_PERCENTAGE')
)

image = collection.first()


def download_image(sub_geom, index):
    tif_file = f'sentinel2_image_part_{index + 1}.tif'
    geemap.ee_export_image(
        image.clip(sub_geom),
        filename=tif_file,
        scale=10,
        region=sub_geom,
        file_per_band=False
    )

with ThreadPoolExecutor(max_workers=4) as executor:  # 4 iş parçası kullanarak
    for i, sub_polygon in enumerate(grid.getInfo()['features']):
        sub_geom = ee.Geometry.Polygon(sub_polygon['geometry']['coordinates'])
        executor.submit(download_image, sub_geom, i)

 #İndirilen parçaları birleştirme
tif_files = glob.glob('sentinel2_image_part_*.tif')

src_files_to_mosaic = [rasterio.open(tif_file) for tif_file in tif_files]

mosaic, out_trans = merge(src_files_to_mosaic)

# Birleştirilmiş görüntü için meta bilgileri güncelleme
out_meta = src_files_to_mosaic[0].meta.copy()
out_meta.update({
    "driver": "GTiff",
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": out_trans
})

# Birleştirilmiş görüntüyü kaydetme
with rasterio.open('mosaic_image.tif', "w", **out_meta) as dest:
    dest.write(mosaic)

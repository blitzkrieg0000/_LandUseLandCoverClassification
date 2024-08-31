import ee
import folium
import geemap.foliumap as geemap
import gradio as gr
from folium import plugins


# Connect GEE API
GEE_CREDENTIALS_FILE = "./data/gee/geospatial_api_key.json"
credentials = ee.ServiceAccountCredentials(None, GEE_CREDENTIALS_FILE)
ee.Initialize(credentials)




def display_images_on_map():
    roi_coords = ee.Geometry.Polygon([[37.279358,36.412414],[37.279358,39.925815],[42.393494,39.925815],[42.393494,36.412414],[37.279358,36.412414]])  # Ankara civarı

    roi = ee.Geometry.Polygon(roi_coords)
    
    collection = ee.ImageCollection('COPERNICUS/S2') \
        .filterBounds(roi) \
        .filterDate("2021-01-01", "2021-02-01") \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    
    map_center = [roi_coords[0][0][1], roi_coords[0][0][0]]
    my_map = folium.Map(location=map_center, zoom_start=10)
    
    for i in range(collection.size().getInfo()):
        img = ee.Image(collection.toList(collection.size()).get(i))
        vis_params = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000}
        url = img.getMapId(vis_params)['tile_fetcher'].url_format
        folium.TileLayer(
            tiles=url,
            attr='Google Earth Engine',
            name=f'Image {i}'
        ).add_to(my_map)
    
    # Poligon çizme aracı ekleyin
    plugins.Draw(
        export=True,
        polygon=True,
        circle=False,
        rectangle=False,
        polyline=False,
        marker=False
    ).add_to(my_map)
    
    return my_map._repr_html_()

def gradio_interface(start_date, end_date):

    return display_images_on_map(start_date, end_date)

# Gradio arayüzü oluşturun
gr.Interface(
    fn=gradio_interface,
    outputs=gr.HTML()
).launch()

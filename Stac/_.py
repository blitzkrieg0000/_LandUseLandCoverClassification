
##! --------------- XYZ Tiles --------------- !##

# Görüntüyü URL olarak alma
url = image.getMapId(viz_params)["tile_fetcher"].url_format

# Folium haritası oluşturma
m = folium.Map(location=[40.7128, -74.0060], zoom_start=10)

# GEE'den alınan veriyi Folium haritasına ekleme
folium.TileLayer(
    tiles=url,
    attr="Google Earth Engine",
    overlay=True,
    name='Sentinel-2'
).add_to(m)

# Haritayı kaydetme
m.save('gee_folium_map.html')



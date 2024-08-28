from ipyleaflet import Map, DrawControl
import json
from ipywidgets import Button, VBox

# Harita oluştur
center = [38.9637, 35.2433]
m = Map(center=center, zoom=6)

# Çizim kontrolü ekle
draw_control = DrawControl()
m.add_control(draw_control)

# Çizim tamamlandığında çalışacak fonksiyon
def handle_draw(target, action, geo_json):
    print(f"GeoJSON data: {geo_json}")
    with open("drawn_shape.geojson", "w") as f:
        json.dump(geo_json, f)
    print("GeoJSON dosyası kaydedildi!")

draw_control.on_draw(handle_draw)

# Haritayı görüntüle


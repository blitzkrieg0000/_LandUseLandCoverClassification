import dash
from dash import dcc, html, Input, Output
import dash_leaflet as dl
import dash_leaflet.express as dlx

# Dash uygulamasını başlat
app = dash.Dash(__name__)

# Uygulama düzenini oluştur
app.layout = html.Div([
    dl.Map(
        children=[
            dl.TileLayer(),
            dl.FeatureGroup(dl.EditControl(
                id="draw-control", 
                draw=dict(polygon=True, circle=True, rectangle=True, marker=True, polyline=True))
            )],
        id="map",
        style={'width': '1000px', 'height': '500px'},
        zoom=5,
        center=[39.8283, -98.5795]
    ),
    html.Div(id='polygon-output')
])

@app.callback(
    Output('polygon-output', 'children'),
    Input('draw-control', 'drawn_items')
)
def display_polygon(drawn_items):
    print(drawn_items)
    if drawn_items and 'features' in drawn_items:
        features = drawn_items['features']
        polygons = [feature for feature in features if feature['geometry']['type'] == 'Polygon']
        return f"Poligonlar: {polygons}"
    return "Poligon bulunamadı."

# Uygulama ana çalıştırma fonksiyonu
if __name__ == '__main__':
    app.run_server(debug=True)

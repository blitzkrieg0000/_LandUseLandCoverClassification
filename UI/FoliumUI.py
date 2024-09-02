# import os
# os.system("pip install folium")
# os.system("pip install gradio")

import folium
import gradio as gr
from folium import plugins



def CreateMap():
    fmap = folium.Map(location=[37.511905, 38.51532], zoom_start=6, world_copy_jump=True, tiles=None)
    
    folium.TileLayer(
        tiles="http://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="OpenStreetMap",
        name="Open Street Map",
    ).add_to(fmap)

    folium.TileLayer(
        tiles="http://www.google.cn/maps/vt?lyrs=s@189&gl=cn&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Satellite",
    ).add_to(fmap)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="EsriNatGeo",
        name="Esri Nat Geo Map",
    ).add_to(fmap)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="EsriWorldStreetMap",
        name="Esri World Street Map",
    ).add_to(fmap)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri World Map",
    ).add_to(fmap)

    folium.TileLayer(
        tiles="https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
        attr="CartoDB",
        name="CartoDB",
    ).add_to(fmap)

    # Plugins
    formatter = "function(num) {return L.Util.formatNum(num, 3) + ' &deg; ';};"
    plugins.MousePosition(
        position="bottomright",
        separator=" | ",
        empty_string="NaN",
        lng_first=True,
        num_digits=20,
        prefix="Koordinatlar:",
        lat_formatter=formatter,
        lng_formatter=formatter,
    ).add_to(fmap)

    plugins.Geocoder().add_to(fmap)
    plugins.Draw(export=True, filename="drawing.geojson", position="topleft", show_geometry_on_click=False).add_to(fmap)
    folium.LayerControl(position="bottomleft",).add_to(fmap)
    plugins.Fullscreen(
        position="topright",
        title="Expand me",
        title_cancel="Exit me",
        force_separate_button=True,
    ).add_to(fmap)

    
    # Drawing Control
    mapObjectInHTML = fmap.get_name()
    root = fmap.get_root()
    root.html.add_child(folium.Element("""
        <script type="text/javascript">
            document.addEventListener('DOMContentLoaded', function() {
                var idx = 0;
                var drawnItems = new L.FeatureGroup();
                {map}.addLayer(drawnItems);
                                       

                const updateGeoJsonOutput = (geoJsonData) => {
                    const textArea = parent.document.querySelector('#geojson_output textarea');
                    textArea.value = JSON.stringify(geoJsonData);
                    textArea.dispatchEvent(new Event('input'));
                };
                                       
                parent.document.getElementById('geojson_process').onclick = function() {
                    updateGeoJsonOutput(drawnItems.toGeoJSON());
                };
                                       
                {map}.on("draw:created", function(e) {           
                    var layer = e.layer;
                    drawnItems.addLayer(layer);
                    feature = layer.feature = layer.feature || {}; 
                    
                    var title, value;
                    do {
                        title = prompt("Şekle bir isim giriniz:", "");
                        if (title === null) {
                            drawnItems.removeLayer(layer);
                            return;
                        }
                    } while (!title);

                    do {
                        value = prompt("Şekle bir değer giriniz: ", "");
                        if (value === null) {
                            drawnItems.removeLayer(layer);
                            return;
                        }
                    } while (!value);              

                    var id = idx++;
                    feature.type = feature.type || "Feature";
                    var props = feature.properties = feature.properties || {};
                    props.Id = id;
                    props.Title = title;
                    props.Value = value;
                                               
                    // Add Tooltip
                    layer.bindTooltip(`Şekil:${id+1} - İsim:${title} - Değer:${value}`, {permanent: false, direction: "center", className: "label-tooltip"}).openTooltip();
                           
                    drawnItems.addLayer(layer);

                    updateGeoJsonOutput(drawnItems.toGeoJSON());
                });
                
                {map}.on('draw:deleted', function(e) {
                    var layers = e.layers;
                    layers.eachLayer(function(layer) {
                        drawnItems.removeLayer(layer);
                        updateGeoJsonOutput(drawnItems.toGeoJSON());
                    });
                });
                                       
            });
        </script>
    """.replace('{map}', mapObjectInHTML)))

    return fmap._repr_html_()


# GeoJSON verisini işleme fonksiyonu
def process_geojson(geojson_data):
    return geojson_data


# Gradio uygulamasını oluşturma
with gr.Blocks() as app:

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Çalışmak İstediğiniz Bölgeyi Çiziniz")

        with gr.Column(scale=1):
            button = gr.Button("Process", elem_id="geojson_process")

        with gr.Column(scale=1):
            uploadGeoJsonButton = gr.Button("GeoJson Yükle", elem_id="geojson_upload")
    

    # Haritayı görüntüleme
    map_html = gr.HTML(CreateMap(), elem_id="map_container")

    # GeoJSON Textbox
    geojson_output = gr.Textbox("", placeholder="Çizilen Şekillerin GeoJSON formatı burada görünür.", label="Raw GeoJSON: ", lines=6, elem_id="geojson_output", interactive=False)
    
    # JSON Prettier GeoJSON Textbox
    geojson_view = gr.JSON(label="PrettierGeoJSON: ", visible=True)


    # Harita ve GeoJSON verisini bağlama
    @button.click(inputs=geojson_output, outputs=geojson_view, scroll_to_output=True)
    def VisualizeAsGeoJson(geojson_data):

        return geojson_data or {}


app.queue(max_size=10)
app.launch(share_server_protocol="https") # auth=("admin", "admin")

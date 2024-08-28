import folium
import gradio as gr
from folium import plugins

def CreateMap():
    m = folium.Map(location=[37.511905, 38.51532], zoom_start=6, world_copy_jump=True, tiles=None)
    
    folium.TileLayer(
        tiles="http://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="OpenStreetMap",
        name="Open Street Map",
    ).add_to(m)

    folium.TileLayer(
        tiles="http://www.google.cn/maps/vt?lyrs=s@189&gl=cn&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Satellite",
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="EsriNatGeo",
        name="Esri Nat Geo Map",
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="EsriWorldStreetMap",
        name="Esri World Street Map",
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri World Map",
    ).add_to(m)

    folium.TileLayer(
        tiles="https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
        attr="CartoDB",
        name="CartoDB",
    ).add_to(m)

    folium.TileLayer(
        tiles="https://tiles.stadiamaps.com/tiles/stamen_toner_background/{z}/{x}/{y}{r}.png",
        attr="stadiamaps",
        name="Stadia Maps",
    ).add_to(m)

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
    ).add_to(m)

    plugins.Geocoder().add_to(m)
    plugins.Draw(export=True, filename="drawing.geojson", position="topleft", show_geometry_on_click=False).add_to(m)
    folium.LayerControl(position="bottomleft",).add_to(m)
    plugins.Fullscreen(
        position="topright",
        title="Expand me",
        title_cancel="Exit me",
        force_separate_button=True,
    ).add_to(m)


    # Drawing Control
    mapObjectInHTML = m.get_name()

    m.get_root().html.add_child(folium.Element("""
        <script type="text/javascript">
            document.addEventListener('DOMContentLoaded', function() {
                var idx = 0;
                                               
                const updateGeoJsonOutput = (geoJsonData) => {
                    const textArea = parent.document.querySelector('#geojson_output textarea');
  
                    textArea.value = JSON.stringify(geoJsonData);
                    textArea.dispatchEvent(new Event('input'));
                };

                {map}.on("draw:created", function(e) {           
                    var layer = e.layer;
                    feature = layer.feature = layer.feature || {}; 
                        
                    var title = prompt("Şekle bir isim giriniz:", "default");
                    var value = prompt("Şekle bir değer giriniz: ", "");
                    var id = idx++;

                    feature.type = feature.type || "Feature";
                    var props = feature.properties = feature.properties || {};
                    props.Id = id;
                    props.Title = title;
                    props.Value = value;
                    drawnItems.addLayer(layer);

                    //updateGeoJsonOutput(e.layer.toGeoJSON());
                });
                                               
                parent.document.getElementById('geojson_process').onclick = function() {
                    updateGeoJsonOutput(drawnItems.toGeoJSON());
                };
            });
        </script>
    """.replace('{map}', mapObjectInHTML)))

    return m._repr_html_()


# GeoJSON verisini işleme fonksiyonu
def process_geojson(geojson_data):
    return geojson_data


# Gradio uygulamasını oluşturma
with gr.Blocks() as app:
    gr.Markdown("## Çalışmak İstediğiniz Bölgeyi Çiziniz")
    
    # Haritayı görüntüleme
    map_html = gr.HTML(CreateMap())
    
    # GeoJSON Textbox
    geojson_output = gr.Textbox("", label="GeoJSON Output", lines=6, elem_id="geojson_output")
    
    # JSON Pretty Textbox
    geojson_view = gr.JSON(visible=True)

    # Process Button
    button = gr.Button("Process", elem_id="geojson_process")

    # Harita ve GeoJSON verisini bağlama
    @button.click(inputs=geojson_output, outputs=geojson_view)
    def VisualizeAsGeoJson(geojson_data):
        return geojson_data or {}
    


app.launch()

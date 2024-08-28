import folium
import gradio as gr
from folium import plugins


def CreateMap():
    m = folium.Map(location=[45.5236, -122.6750], zoom_start=13)
    
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="World Topo Map"
    ).add_to(m)

    plugins.Geocoder().add_to(m)

    # Çizim kontrolü ekle
    plugins.Draw(export=True, filename="drawing.geojson", position="topleft").add_to(m)
    
    mapObjectInHTML = m.get_name()

    m.get_root().html.add_child(folium.Element("""
        <script type="text/javascript">
            document.addEventListener('DOMContentLoaded', function(){             
                {map}.on("draw:create", function(e){
                    console.log("on event");
                    const geoJsonData = JSON.stringify(e.layer.toGeoJSON());
                    console.log(geoJsonData);
                                                                                        
                    var textArea = parent.document.querySelector('#geojson_output textarea');
                    textArea.value = geoJsonData;
                    var event = new Event('input');
                    textArea.dispatchEvent(event);
                });    
            
                parent.document.getElementById('geojson_process').onclick = function(e) {
                    var data = drawnItems.toGeoJSON();
                    //var convertedData = 'text/json;charset=utf-8,'+ encodeURIComponent(JSON.stringify(data));
                    var convertedData = JSON.stringify(data);                       
                    
                    var textArea = parent.document.querySelector('#geojson_output textarea');
                    textArea.value = convertedData;
                    var event = new Event('input');
                    textArea.dispatchEvent(event);
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
    geojson_output = gr.Textbox("", label="GeoJSON Output", lines=10, elem_id="geojson_output")
    
    # JSON Pretty Textbox
    geojson_view = gr.JSON(visible=True)

    # Process Button
    button = gr.Button("Process", elem_id="geojson_process")

    # Harita ve GeoJSON verisini bağlama
    @button.click(inputs=geojson_output, outputs=geojson_view)
    def VisualizeAsGeoJson(geojson_data):
        return geojson_data or {}
    


app.launch()

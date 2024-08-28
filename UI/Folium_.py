import folium
from folium import plugins
import gradio as gr

# Harita oluşturma fonksiyonu
def create_map():
    m = folium.Map(location=[45.5236, -122.6750], zoom_start=13)

    # Çizim kontrolü ekle
    plugins.Draw(export=True, filename='drawing.geojson', position = 'topleft').add_to(m)

    mapObjectInHTML = m.get_name()

    m.get_root().html.add_child(folium.Element("""
        <script type="text/javascript">
                                                
            document.addEventListener('DOMContentLoaded', function(){             
                {map}.on("draw:created", function(e){
                                                
                    const geoJsonData = JSON.stringify(e.layer.toGeoJSON());
                    console.log(geoJsonData);
                                                                                        
                    var textArea = parent.document.querySelector('#geojson_output textarea');
                    textArea.value = geoJsonData

                });    
            });
                                                    
        </script>
    """.replace('{map}', mapObjectInHTML)))

    return m._repr_html_()

# GeoJSON verisini işleme fonksiyonu
def process_geojson(geojson_data):
    return geojson_data


# Gradio uygulamasını oluşturma
with gr.Blocks() as app:
    gr.Markdown("## Harita üzerinde çizilen poligondan GeoJSON al")
    
    # Haritayı görüntüleme
    map_html = gr.HTML(create_map())
    
    
    # GeoJSON verisini göstermek için bir textbox
    geojson_output = gr.Textbox(label="GeoJSON Output", lines=10, elem_id="geojson_output", elem_classes="geojson_output")
    

    # Harita ve GeoJSON verisini bağlama
    geojson_input = gr.Textbox(visible=False)
    geojson_input.change(process_geojson, inputs=geojson_input, outputs=geojson_output)



app.launch()

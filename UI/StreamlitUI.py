import json
import folium
import streamlit as st
from folium import plugins
from streamlit_folium import st_folium, folium_static
import streamlit.components.v1 as components

from lib import RequestFunction
st.set_page_config(layout="wide")


def CreateMap():
    fmap = folium.Map(location=[38.534546, 37.509726], crs="EPSG3857", zoom_start=6, world_copy_jump=True, tiles=None)

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
    plugins.Draw(export=False, filename="drawing.geojson", position="topleft", show_geometry_on_click=False).add_to(fmap)
    folium.LayerControl(position="bottomleft",).add_to(fmap)
    plugins.Fullscreen(
        position="topright",
        title="Expand me",
        title_cancel="Exit me",
        force_separate_button=True,
    ).add_to(fmap)
    
    return fmap



def CreateDatasetMap():
    st.session_state["activate_map"] = True
    geojson = st.session_state["fmap_output"]
    coodinates = geojson.get("all_drawings")[0].get("geometry").get("coordinates")[0]

    data = RequestFunction(coodinates)
    st.session_state["dataset_map"] = data
    with datasetMapContainer:
        st.write("## Dataset Map")
        st_folium(data)


def ResetDatasetMap():
    st.session_state["activate_map"] = None


fmap = CreateMap()


##! --------------- Render --------------- !##
mapContainer = st.container(border=True)
st.sidebar.write("## LULC Data Visualizer")


with mapContainer:
    st.write("## LULC Data Explorer")
    st.write("##### Çalışmak istediğiniz bölgeye ait bir şekil çiziniz ve ardından yapmak istediğiniz işlemi seçiniz.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write("### İşlemler:")

    with col2:
        saveGeoJsonButton = st.button("Get LULC Map", on_click=CreateDatasetMap, use_container_width=True)

    with col3:
        showGeoJsonButton = st.button("Show GeoJson", use_container_width=True)

    with col4:
        resetDatasetMapButton = st.button("Reset", on_click=ResetDatasetMap, use_container_width=True)


    output = st_folium(fmap, use_container_width=True)
    st.session_state["fmap_output"] = output
    

# Show GeoJson
if showGeoJsonButton:
    st.json(st.session_state["fmap_output"], expanded=False)


# Result Container
datasetMapContainer = st.container(border=True)

if st.session_state.get("dataset_map"):
    with datasetMapContainer:
        st.write("## Dataset Map")
        folium_static(st.session_state.get("dataset_map"))
import os
import ee
import geemap.foliumap as geemap
import gradio as gr
from matplotlib import pyplot as plt
import requests

# geemap.update_package()

# Connect GEE API
GEE_CREDENTIALS_FILE = "./data/gee/geospatial_api_key.json"
if not os.path.exists(GEE_CREDENTIALS_FILE):
    credentials = ee.ServiceAccountCredentials(None, key_data=os.getenv("geospatial_api_key"))
else:
    credentials = ee.ServiceAccountCredentials(None, GEE_CREDENTIALS_FILE)

ee.Initialize(credentials)


##! --------------- PARAMS --------------- !##
ROI = ee.Geometry.Polygon([[37.279358,36.412414],[37.279358, 39.925815],[42.393494, 39.925815],[42.393494, 36.412414],[37.279358, 36.412414]])

CLASSES = {
    "names" : ["Water", "Trees", "Flooded Vegetation", "Crops", "Built Area", "Bare Ground", "Snow/Ice", "Clouds", "Rangeland"],
    "colors" : ["1a5bab", "358221", "87d19e", "ffdb5c", "ed022a", "ede9e4", "f2faff", "c8c8c8", "c6ad8d"]
}
CLOUD_FILTER = 60
CLD_PRB_THRESH = 50
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 1
BUFFER = 50
CLOUDY_PIXEL_PERCENTAGE = 12
S2_START_DATE = "2023-08-01"
S2_END_DATE = "2023-08-31"
S2_LULC_START_DATE = "2023-01-01"
S2_LULC_END_DATE = "2023-12-31"

S2_VIS_PARAMS = {
    "bands": ["B4", "B3", "B2"],
    "min": 0,
    "max": 3000,
    "gamma": [1, 1, 1],
}




##! --------------- Functions --------------- !##
def VisualizeMatplot(image, roi):
    """Görüntüyü numpy array olarak alma"""
    image = geemap.ee_to_numpy(image, region=roi, bands=["B4", "B3", "B2"])

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def GMap2Gradio(image, vis_params):
    # Görüntüyü RGB bantları ile görselleştirme
    vis_params = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}
    gMap = geemap.Map()
    gMap.add_layer(image, vis_params, "Sentinel-2 Image")
    return gMap.to_gradio()


def ApplyBasicS2CloudMask(image):
    cloud_mask = image.select("MSK_CLASSI_CIRRUS").lt(1)
    return image.updateMask(cloud_mask)


def ApplyBitwiseS2CloudMask(image):
    qa = image.select("QA60")

    # QA Bandının Bit 10 ve 11 bulut ve bulut gölgeleri için kullanılır
    cloudBitMask = 1 << 10

    cirrusBitMask = 1 << 11

    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))

    return image.updateMask(mask) #.divide(10000)


def ReColormap(image, old_colors: list, new_colors: list):
    return image.remap(old_colors, new_colors)


def CalculateRasterOverlapByROI(image, roi: ee.Geometry):
    intersection = image.clip(roi).reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=roi,
        scale=10,
        maxPixels=1e9
    )
    count = ee.Number(intersection.get("B4"))  # B4 bandı kullanılıyor
    return image.set("overlap", count)


def GetS2SRCloudCollection(s2_collection: ee.ImageCollection, roi: ee.Geometry, start_date: str, end_date: str):
    s2_cloudless_col = (ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(roi)
        .filterDate(start_date, end_date))

    return ee.ImageCollection(ee.Join.saveFirst("s2cloudless")
        .apply(
            primary=s2_collection,
            secondary=s2_cloudless_col,
            condition=ee.Filter.equals(leftField="system:index", rightField="system:index")
        )
    )



# Her bir görüntü için alanı hesapla ve NoData piksellerini dışla
def add_area_attribute(image, roi):
    # Belirli geometri içindeki piksel alanlarını hesapla
    pixel_area = image.select(0).multiply(ee.Image.pixelArea())
    
    # Hesaplanan alanı, istenen geometri içinde özetle
    region_area = pixel_area.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi,
        scale=10,         # Sentinel-2 için 10 metre çözünürlük
        maxPixels=1e13
    ).get("B1")  # Burada B1 rastgele bir banttır. İlgili bantlardan biri kullanılabilir
    
    # Yeni bir attribute olarak alanı ekle
    return image.set("REGION_AREA", region_area)


def CalculateIntersectionArea(image, roi):
    # Görüntüyü maskele ve konveks kaplamasını hesapla
    convex_hull = image.geometry().convexHull()
    
    # Konveks kaplamanın region ile kesişimini hesapla
    intersection = convex_hull.intersection(roi)
    
    # Kesişimin alanını hesapla
    intersection_area = intersection.area()
    
    # Yeni bir attribute olarak kesişim alanını ekle
    return image.set("INTERSECTION_AREA", intersection_area)



def FilterBestAreaOfMGRSJoin(image):
    return ee.ImageCollection.fromImages(
        image.get("mgrs_tile_match")
    ).sort("INTERSECTION_AREA", False).first()



def RequestFunction(roi):
    if roi is None:
        return None

    roi = ee.Geometry.Polygon(roi)

    ##! --------------- Dataset --------------- !##
    # Gather S2 Data
    S2Collection = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filterBounds(roi)
            .filterDate(S2_START_DATE, S2_END_DATE)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUDY_PIXEL_PERCENTAGE))
            .sort("CLOUDY_PIXEL_PERCENTAGE", ascending=True)
    )

    # Gather Mask Data
    esriLULCCollection = ee.ImageCollection("projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS").filterDate(S2_LULC_START_DATE, S2_LULC_END_DATE)
    esriLULCCollection = (
        ee.ImageCollection(esriLULCCollection.mosaic().clip(roi))
        .map(lambda img: ReColormap(img, [1, 2, 4, 5, 7, 8, 9, 10, 11], [1, 2, 3, 4, 5, 6, 7, 8, 9]))
    )

    #! Get Info
    image_info = S2Collection.first().getInfo()
    print(list(zip(list(image_info["properties"].keys()), list(image_info["properties"].values()))))
    print(image_info["bands"])

    # Cloud Masking
    S2Collection = S2Collection.map(ApplyBitwiseS2CloudMask)

    # Area Calculation
    S2Collection = S2Collection.map(lambda image: CalculateIntersectionArea(image, roi)).sort("INTERSECTION_AREA", False)
    
    # MGRS_TILE Match
    joinCol = ee.Join.saveAll("mgrs_tile_match").apply(
        S2Collection.distinct("MGRS_TILE"),
        S2Collection,
        ee.Filter.equals(leftField="MGRS_TILE", rightField="MGRS_TILE")
    )

    bestCollection = ee.ImageCollection(joinCol.map(FilterBestAreaOfMGRSJoin))
    s2Image = bestCollection.mosaic().clip(roi)
    # print(bestCollection.aggregate_array("INTERSECTION_AREA").getInfo())


    #! --------------- Mask Match --------------- !##
    esriLULCImage = esriLULCCollection.first().updateMask(s2Image.select("B4").gt(0))


    ##! --------------- Create FoliumGee Map --------------- !##
    map = geemap.Map(center=[37.279358, 36.412414], zoom=10)
    map.add_layer(roi, {"color": "black"}, "Pilot Bölge")
    map.add_layer(s2Image, S2_VIS_PARAMS, "RGB")
    map.add_layer(roi, {"color": "black"}, "Pilot Bölge")
    map.add_layer(esriLULCImage, {"min": 1, "max": 9, "palette": CLASSES["colors"]}, "LULC")


    return map.to_gradio()
    # url = esriLULCImage.getDownloadURL({
    #     "scale": 10,  # Çözünürlük
    #     "region": ROI,
    #     "format": "GeoTIFF"
    # })



if "__main__" == __name__:

    html = RequestFunction(ROI)

    with gr.Blocks() as app:
        web = gr.HTML(html, label="Map")

    app.launch()

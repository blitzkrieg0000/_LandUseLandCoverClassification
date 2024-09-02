import ee
import geemap.foliumap as geemap
import gradio as gr
from matplotlib import pyplot as plt

# Connect GEE API
GEE_CREDENTIALS_FILE = "./data/gee/geospatial_api_key.json"
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



##! --------------- Functions --------------- !##
def VisualizeMatplot(image):
    """Görüntüyü numpy array olarak alma"""
    image = geemap.ee_to_numpy(image, region=ROI, bands=["B4", "B3", "B2"])

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


def CalculateRasterOverlapByROI(image):
    intersection = image.clip(ROI).reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=ROI,
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

# Bir görüntüyü maskele
def mask_nodata(image):
    return image.updateMask(image)

# Her bir görüntü için alanı hesapla ve NoData piksellerini dışla
def add_area_attribute(image):
    # NoData olan pikselleri maskelenmiş görüntü
    masked_image = mask_nodata(image)
    
    # Belirli geometri içindeki piksel alanlarını hesapla
    pixel_area = masked_image.select(0).multiply(ee.Image.pixelArea())
    
    # Hesaplanan alanı, istenen geometri içinde özetle
    region_area = pixel_area.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=ROI,
        scale=10,         # Sentinel-2 için 10 metre çözünürlük
        maxPixels=1e13
    ).get("B1")  # Burada B1 rastgele bir banttır. İlgili bantlardan biri kullanılabilir
    
    # Yeni bir attribute olarak alanı ekle
    return image.set("REGION_AREA", region_area)


def calculate_intersection_area(image):
    # Görüntüyü maskele ve konveks kaplamasını hesapla
    convex_hull = image.geometry().convexHull()
    
    # Konveks kaplamanın region ile kesişimini hesapla
    intersection = convex_hull.intersection(ROI)
    
    # Kesişimin alanını hesapla
    intersection_area = intersection.area()
    
    # Yeni bir attribute olarak kesişim alanını ekle
    return image.set("INTERSECTION_AREA", intersection_area)


def FilterBestAreaOfMGRSJoin(image):
    bestTile = ee.ImageCollection.fromImages(image.get("mgrs_tile_match")).sort("INTERSECTION_AREA", False).first()
    return bestTile




def Main():
    collection = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filterBounds(ROI)
            .filterDate("2024-08-01", "2024-08-31")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 12))
            .sort("CLOUDY_PIXEL_PERCENTAGE", ascending=True)
            
    )
    
    # Cloud Masking
    collection = collection.map(ApplyBitwiseS2CloudMask)
    
    
    esri_lulc10 = ee.ImageCollection("projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS")
    lulc_image = ee.ImageCollection(esri_lulc10.filterDate("2023-01-01","2023-12-31").mosaic().clip(ROI)).map(lambda img: ReColormap(img, [1, 2, 4, 5, 7, 8, 9, 10, 11], [1, 2, 3, 4, 5, 6, 7, 8, 9]))


    # Define the visualization parameters.
    viz_params = {
        "bands": ["B4", "B3", "B2"],
        "min": 0,
        "max": 3000,
        "gamma": [1, 1, 1],
    }


    sentinel2_with_area = collection.map(CalculateRasterOverlapByROI)

    # Sort images by area covered
    sorted_images = sentinel2_with_area.sort("area", False)

    image = sorted_images.mosaic().clip(ROI)

    # Define a map centered on San Francisco Bay.
    map = geemap.Map(center=[37.279358, 36.412414], zoom=10)
    map.add_layer(ROI, {"color": "black"}, "Pilot Bölge")
    map.add_layer(lulc_image, {"min": 1, "max": 9, "palette": CLASSES["colors"]}, "LULC")
    map.add_layer(image, viz_params, "RGB")

    legend_dict = {name: color for name, color in zip(CLASSES["names"], CLASSES["colors"])}
    legend = geemap.create_legend(title="Legend", legend_dict=legend_dict, draggable=False, position="bottomright")


    with gr.Blocks() as app:
        web = gr.HTML(map.to_gradio(), label="Map")


    app.launch()

    return 0


def Main2():
    collection = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filterBounds(ROI)
            .filterDate("2024-08-01", "2024-08-31")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 12))
            .sort("CLOUDY_PIXEL_PERCENTAGE", ascending=True)
    )

    #! Get Info
    image_info = collection.first().getInfo()
    print(list(zip(list(image_info["properties"].keys()), list(image_info["properties"].values()))))
    print(image_info["bands"])

    #! Tüm koleksiyona yeni attribute'u ekle
    collection_with_area = collection.map(calculate_intersection_area).sort("INTERSECTION_AREA", False)
    

    # MGRS_TILE
    distinctTiles = collection.distinct("MGRS_TILE")
    filter = ee.Filter.equals(leftField="MGRS_TILE", rightField="MGRS_TILE")
    join = ee.Join.saveAll("mgrs_tile_match")
    joinCol = join.apply(distinctTiles, collection_with_area, filter)
    bestCollection = ee.ImageCollection(joinCol.map(FilterBestAreaOfMGRSJoin))
    print(bestCollection.aggregate_array("INTERSECTION_AREA").getInfo())

    viz_params = {
        "bands": ["B4", "B3", "B2"],
        "min": 0,
        "max": 3000,
        "gamma": [1, 1, 1],
    }


    #! Visualize
    image = bestCollection.mosaic().clip(ROI)
    esri_lulc10 = ee.ImageCollection("projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS")
    lulc_image = ee.ImageCollection(esri_lulc10.filterDate("2023-01-01","2023-12-31").mosaic().clip(ROI)).map(lambda img: ReColormap(img, [1, 2, 4, 5, 7, 8, 9, 10, 11], [1, 2, 3, 4, 5, 6, 7, 8, 9]))
    map = geemap.Map(center=[37.279358, 36.412414], zoom=10)
    map.add_layer(ROI, {"color": "black"}, "Pilot Bölge")
    map.add_layer(image, viz_params, "RGB")
    map.add_layer(ROI, {"color": "black"}, "Pilot Bölge")
    map.add_layer(lulc_image, {"min": 1, "max": 9, "palette": CLASSES["colors"]}, "LULC")


    with gr.Blocks() as app:
        web = gr.HTML(map.to_gradio(), label="Map")

    app.launch()


if "__main__" == __name__:
    # Main()
    Main2()

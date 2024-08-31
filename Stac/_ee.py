import ee
import geemap.foliumap as geemap
import gradio as gr
from matplotlib import pyplot as plt

# Connect GEE API
GEE_CREDENTIALS_FILE = "./data/gee/geospatial_api_key.json"
credentials = ee.ServiceAccountCredentials(None, GEE_CREDENTIALS_FILE)
ee.Initialize(credentials)



##! --------------- PARAMS --------------- !##
region = ee.Geometry.Polygon([[37.279358,36.412414],[37.279358,39.925815],[42.393494,39.925815],[42.393494,36.412414],[37.279358,36.412414]])  # Ankara civarı
classes = {
    "names": ["Water", "Trees", "Flooded Vegetation", "Crops", "Built Area", "Bare Ground", "Snow/Ice", "Clouds", "Rangeland"],
    "colors": ["1a5bab", "358221", "87d19e", "ffdb5c", "ed022a", "ede9e4", "f2faff", "c8c8c8", "c6ad8d"]
}


##! --------------- Functions --------------- !##
def VisualizeMatplot(image):
    #! 2- Görüntüyü numpy array olarak alma
    rgb_image = geemap.ee_to_numpy(image, region=region, bands=["B4", "B3", "B2"])
    # Görüntüyü çizdirme
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image)
    plt.axis("off")
    plt.show()


def GMap2Gradio(image):
    # Görüntüyü RGB bantları ile görselleştirme
    vis_params = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}
    gMap = geemap.Map()
    gMap.add_ee_layer(image, vis_params, "Sentinel-2 Image")
    return gMap.to_gradio()


def MaskClouds(image):
    cloud_mask = image.select("MSK_CLASSI_CIRRUS").lt(1)
    return image.updateMask(cloud_mask)


def MaskS2Clouds(image):
    qa = image.select("QA60")

    # Bit 10 ve 11 bulut ve bulut gölgeleri için kullanılır
    cloudBitMask = ee.Number(2).pow(10).int()
    cirrusBitMask = ee.Number(2).pow(11).int()

    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))

    return image.updateMask(mask) #.divide(10000)


def Remapper(image):
    remapped = image.remap([1, 2, 4, 5, 7, 8, 9, 10, 11], [1, 2, 3, 4, 5, 6, 7, 8, 9])
    return remapped


def CalculateOverlap(image):
    intersection = image.clip(region).reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=region,
        scale=10,
        maxPixels=1e9
    )
    count = ee.Number(intersection.get("B4"))  # B4 bandı kullanılıyor
    return image.set("overlap", count)


def Main():
    collection = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filterBounds(region)
            .filterDate("2024-08-01", "2024-08-31")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 12))
            .sort("CLOUDY_PIXEL_PERCENTAGE", ascending=True)
            
    )
    
    # Cloud Masking
    collection = collection.map(MaskS2Clouds)
    
    
    esri_lulc10 = ee.ImageCollection("projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS")
    lulc_image = ee.ImageCollection(esri_lulc10.filterDate("2023-01-01","2023-12-31").mosaic().clip(region)).map(Remapper)


    # Define the visualization parameters.
    viz_params = {
        "bands": ["B4", "B3", "B2"],
        "min": 0,
        "max": 3000,
        "gamma": [1, 1, 1],
    }


    sentinel2_with_area = collection.map(CalculateOverlap)

    # Sort images by area covered
    sorted_images = sentinel2_with_area.sort("area", False)

    image = sorted_images.mosaic().clip(region)

    # Define a map centered on San Francisco Bay.
    map = geemap.Map(center=[37.279358, 36.412414], zoom=10)
    map.add_layer(region, {"color": "black"}, "Pilot Bölge")
    map.add_layer(lulc_image, {"min": 1, "max": 9, "palette": classes["colors"]}, "LULC")
    map.add_layer(image, viz_params, "RGB")

    legend_dict = {name: color for name, color in zip(classes["names"], classes["colors"])}
    legend = geemap.create_legend(title="Legend", legend_dict=legend_dict, draggable=False, position="bottomright")


    with gr.Blocks() as app:
        web = gr.HTML(map.to_gradio(), label="Map")


    app.launch()

    return 0


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
        geometry=region,
        scale=10,         # Sentinel-2 için 10 metre çözünürlük
        maxPixels=1e13
    ).get("B1")  # Burada B1 rastgele bir banttır. İlgili bantlardan biri kullanılabilir
    
    # Yeni bir attribute olarak alanı ekle
    return image.set("REGION_AREA", region_area)


def calculate_intersection_area(image):
    # Görüntüyü maskele ve konveks kaplamasını hesapla
    convex_hull = image.geometry().convexHull()
    
    # Konveks kaplamanın region ile kesişimini hesapla
    intersection = convex_hull.intersection(region)
    
    # Kesişimin alanını hesapla
    intersection_area = intersection.area()
    
    # Yeni bir attribute olarak kesişim alanını ekle
    return image.set("INTERSECTION_AREA", intersection_area)


def FilterBestAreaOfMGRSJoin(image):
    bestTile = ee.ImageCollection.fromImages(image.get("mgrs_tile_match")).sort("INTERSECTION_AREA", False).first()
    return bestTile


if "__main__" == __name__:
    # Main()

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(region)
            .filterDate("2024-08-01", "2024-08-31")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 12))
            .sort("CLOUDY_PIXEL_PERCENTAGE", ascending=True)
    )

    #! Get Info
    # image_info = collection.first().getInfo()
    # print(list(zip(list(image_info["properties"].keys()), list(image_info["properties"].values()))))


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
    image = bestCollection.mosaic().clip(region)
    esri_lulc10 = ee.ImageCollection("projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS")
    lulc_image = ee.ImageCollection(esri_lulc10.filterDate("2023-01-01","2023-12-31").mosaic().clip(region)).map(Remapper)
    map = geemap.Map(center=[37.279358, 36.412414], zoom=10)
    map.add_layer(region, {"color": "black"}, "Pilot Bölge")
    map.add_layer(image, viz_params, "RGB")
    map.add_layer(region, {"color": "black"}, "Pilot Bölge")
    map.add_layer(lulc_image, {"min": 1, "max": 9, "palette": classes["colors"]}, "LULC")
    with gr.Blocks() as app:
        web = gr.HTML(map.to_gradio(), label="Map")

    app.launch()
import json


geojson = '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{"Id":0,"Title":"2","Value":"2"},"geometry":{"type":"Polygon","coordinates":[[[30.938943,37.45302],[30.938943,41.176559],[39.4658,41.176559],[39.4658,37.45302],[30.938943,37.45302]]]}}]}'

geojson = json.loads(geojson)
coodinates = geojson.get("features")[0].get("geometry").get("coordinates")[0]
print(coodinates)
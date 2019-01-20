import copy
import fiona
import json

# slice_number,year,jday (похоже, при создании срезов была ошибка: последний срез в году составил всего 2 дня :((. Поэтому в нумерации произошел сдвиг. Но даже если это не так, то лучше ошибиться в дате и прописать более позднюю дату, чем более раннюю)
lookup = {
        0: (2016,316),
        1: (2016,332),
        2: (2016,348),
        3: (2016,364),
        4: (2016,366),
        5: (2017,16),
        6: (2017,32),
        7: (2017,48),
        8: (2017,64),
        9: (2017,80),
        10: (2017,96),
        11: (2017,112),
        12: (2017,128),
        13: (2017,144),
        14: (2017,160),
}



features = []


with fiona.collection('merged.geojson') as source:
    for feat in source:
        slice = int(feat['properties']['slice'] )
        if slice >= 0:
            dates = [lookup[slice]]
        else:
           # import ipdb; ipdb.set_trace()
           dates = [lookup[k]  for k in lookup.keys()]
        for year, day in dates:
            f = copy.deepcopy(feat)
            f['properties']['year'] = year
            f['properties']['jday'] = day
            features.append(f)

    crs = ' '.join("+%s=%s" % (k,v) for k,v in source.crs.items())

my_layer = {
    "type": "FeatureCollection",
    "features": features,
    "crs": {
        "type": "link", 
        "properties": {"href": "my_layer.crs", "type": "proj4"} }}


with open("my_layer.json", "w") as f:
    f.write(json.dumps(my_layer))
with open("my_layer.crs", "w") as f:
    f.write(crs)

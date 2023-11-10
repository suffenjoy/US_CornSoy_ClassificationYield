var IA = ee.FeatureCollection("projects/suffenjoy7/assets/csb_IA_2022");
//Extract time series from CSB field boundaries
var states = ee.FeatureCollection('TIGER/2016/States');
// var counties = ee.FeatureCollection('TIGER/2016/Counties');


var Iowa = states.filter(ee.Filter.eq('NAME', 'Iowa'));

//var dataset = ee.FeatureCollection('TIGER/2016/Counties');
print(Iowa, "Iowa");

function maskS2clouds(image) {
  var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask).divide(10000);
}

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterDate('2022-01-01', '2022-07-01')
                  .filterBounds(Iowa)
                  // Pre-filter to get less cloudy granules.
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20))
                  .map(maskS2clouds)
                  .select(['B2','B3','B4','B5','B6','B7','B8','B11','B12']);
                  
                  
print(s2.first(), "s2 first image");


//var shapefile = IL;
var shapefile = IA;
Map.addLayer(shapefile);
print(shapefile.limit(10), "shapefile example");
print(shapefile.size(), "shapefile size");

var s2_ts = s2.map(function(image) {
    return image.reduceRegions({
//  return image.select('B2').reduceRegions({
    collection: shapefile, 
    reducer: ee.Reducer.mean(), 
    scale: 20
  }).filter(ee.Filter.neq('B2', null)).filter(ee.Filter.neq('B3', null)).filter(ee.Filter.neq('B4', null)).filter(ee.Filter.neq('B5', null)).filter(ee.Filter.neq('B6', null)).filter(ee.Filter.neq('B7', null)).filter(ee.Filter.neq('B8', null)).filter(ee.Filter.neq('B11', null)).filter(ee.Filter.neq('B12', null))
  .map(function(f) { 
    return f.set('imageId', image.id());
  });
}).flatten();

var s2_ts = s2_ts.select(['system:index','CNTY','CSBACRES','CSBID','B2','B3','B4','B5','B6','B7','B8','B11','B12']);
print(s2_ts.limit(2), "s2_ts example");

var s2_ts = s2_ts.map(function(feature) {
    var dict = feature.toDictionary();
    return ee.Feature(null, dict);
});
print(s2_ts.limit(2), "s2_ts example");

Export.table.toDrive({
  collection: s2_ts,
  description: 's2_ts_2022_IA',
  folder: 'CSB_TimeSeries',
  fileFormat: 'CSV'
});



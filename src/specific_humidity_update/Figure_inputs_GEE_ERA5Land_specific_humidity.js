/******************************************************************************
 * Monthly ERA5-Land SPECIFIC HUMIDITY (kg/kg) over British Columbia, 2000-2024
 * FINAL - BC name in GAUL is the bilingual 'British Columbia / Colombie-Britannique'
 ******************************************************************************/

// ---- REGION: British Columbia (correct bilingual GAUL name) --------------
var bc = ee.FeatureCollection('FAO/GAUL/2015/level1')
           .filter(ee.Filter.eq('ADM1_NAME', 'British Columbia / Colombie-Britannique'))
           .geometry();

print('CHECK - BC area (km2), should be ~944000:', bc.area(1).divide(1e6));
Map.centerObject(bc, 4); Map.addLayer(bc, {color:'red'}, 'BC');

// ---- ERA5-Land + specific humidity (from dewpoint + surface pressure) -----
var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
             .filterDate('2000-01-01', '2025-01-01');

function addSH(img) {
  var Td = img.select('dewpoint_temperature_2m').subtract(273.15);          // K -> degC
  var e  = Td.expression('6.112 * exp(17.67 * b / (b + 243.5))', {b: Td});  // hPa
  var P  = img.select('surface_pressure').divide(100.0);                    // Pa -> hPa
  var q  = ee.Image(0.622).multiply(e).divide(P.subtract(e.multiply(0.378)))
             .rename('specific_humidity');
  var d = ee.Date(img.get('system:time_start'));
  return q.set('year', d.get('year')).set('month', d.get('month'));
}
var qCol = era5.map(addSH);

print('CHECK - test q, first month (should be ~0.003-0.006):',
      qCol.first().reduceRegion({reducer: ee.Reducer.mean(), geometry: bc,
                                 scale: 11132, maxPixels: 1e13}));

// ---- monthly province-mean table + export to Drive -----------------------
var table = qCol.map(function(img) {
  var m = img.reduceRegion({reducer: ee.Reducer.mean(), geometry: bc,
                            scale: 11132, maxPixels: 1e13});
  return ee.Feature(null, {year: img.get('year'), month: img.get('month'),
                           specific_humidity_mean: m.get('specific_humidity')});
});
Export.table.toDrive({
  collection:  ee.FeatureCollection(table),
  description: 'BC_ERA5Land_monthly_specific_humidity_2000_2024',
  fileFormat:  'CSV',
  selectors:   ['year', 'month', 'specific_humidity_mean']
});

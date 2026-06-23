# `data/rasters/` — conditioning-factor rasters

The 18 gridded predictors used by the model, each a GeoTIFF on the **1.5 km analysis grid
(EPSG:3005)**. Native source resolutions and providers are in `../../DATA_SOURCES.md`.

- **Topographic:** `Elevation.tif`, `Slope.tif`, `Aspect.tif`, `Plan_Curvature.tif`,
  `Profile_Curvature.tif`, `TWI.tif`
- **Vegetative:** `NDVI.tif`, `LULC.tif`
- **Climatic:** `Max_Temperature.tif`, `Precipitation.tif`, `WS.tif` (wind speed),
  `Specific_Humidity.tif`, `AET.tif`, `DSI.tif`, `Soil_Moisture.tif`
- **Anthropogenic:** `Distance_roads.tif`, `Distance_rivers.tif`, `Distance_households.tif`

Per-layer ISO 19115-2 metadata: `../../metadata/iso19115/`.

import geopandas as gpd
import os

dir_vec = r"D:\US_CornSoy_ClassificationYield\Geodata\Vectors"
csb_22 = gpd.read_file(r"D:\US_CornSoy_ClassificationYield\Geodata\2022_National_CSB_gdb\CSB1522.gdb", layer = 'nationalGIS')

# Subset based on STATEFIPS
## Iowa
csb_IA_22 = csb_22[csb_22['STATEFIPS'] == '19']
csb_IA_22.to_file(os.path.join(dir_vec, 'csb_IA_2022.shp'))

## Illinois
csb_IL_22 = csb_22[csb_22['STATEFIPS'] == '17']
csb_IL_22.to_file(os.path.join(dir_vec, 'csb_IL_2022.shp'))

## Nebraska
csb_NE_22 = csb_22[csb_22['STATEFIPS'] == '31']
csb_NE_22.to_file(os.path.join(dir_vec, 'csb_NE_2022.shp'))

## Minnesota
csb_MN_22 = csb_22[csb_22['STATEFIPS'] == '27']
csb_MN_22.to_file(os.path.join(dir_vec, 'csb_MN_2022.shp'))

## Indiana
csb_IN_22 = csb_22[csb_22['STATEFIPS'] == '18']
csb_IN_22.to_file(os.path.join(dir_vec, 'csb_IN_2022.shp'))

## Ohio
csb_OH_22 = csb_22[csb_22['STATEFIPS'] == '39']
csb_OH_22.to_file(os.path.join(dir_vec, 'csb_OH_2022.shp'))

## South Dakota
csb_SD_22 = csb_22[csb_22['STATEFIPS'] == '46']
csb_SD_22.to_file(os.path.join(dir_vec, 'csb_SD_2022.shp'))

## Missouri
csb_MO_22 = csb_22[csb_22['STATEFIPS'] == '29']
csb_MO_22.to_file(os.path.join(dir_vec, 'csb_MO_2022.shp'))

## Wisconsin
csb_WI_22 = csb_22[csb_22['STATEFIPS'] == '55']
csb_WI_22.to_file(os.path.join(dir_vec, 'csb_WI_2022.shp'))

## Kansas
csb_KS_22 = csb_22[csb_22['STATEFIPS'] == '20']
csb_KS_22.to_file(os.path.join(dir_vec, 'csb_KS_2022.shp'))

## Michigan
csb_MI_22 = csb_22[csb_22['STATEFIPS'] == '26']
csb_MI_22.to_file(os.path.join(dir_vec, 'csb_MI_2022.shp'))

## Kentucky
csb_KY_22 = csb_22[csb_22['STATEFIPS'] == '21']
csb_KY_22.to_file(os.path.join(dir_vec, 'csb_KY_2022.shp'))
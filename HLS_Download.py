import os, sys
import argparse
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
import shutil
import json 
from glob import glob
from osgeo import gdal
import numpy as np
from rasterstats import zonal_stats
import pdb
import time
from csv import writer

def get_tiles(shpfl):
    shp = gpd.read_file(shpfl)
    shp = shp.to_crs('epsg:4326')
    xmin, ymin, xmax, ymax = shp.total_bounds
    bounds = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])
    mgs = gpd.read_file(r"C:\Users\tangz\Desktop\Base_Data\Sentinel-2-Shapefile-Index\sentinel_2_index_shapefile.shp")
    # intersect with MGS
    mgs_intersect = mgs[mgs.intersects(bounds)]
    # get tile names 
    tiles = mgs_intersect['Name'].values.tolist()
    print('Tiles to download: {}'.format(len(tiles)))
    return tiles

def list_hls(sensors,sdate,edate,tiles):
    """Function downloading HLS Image in batch;
    itype: dtype     --'S10','L30','S30';
           strtime   --acquired starting time of data 'YYYYMMDD'
           endtime   --acquired ending time of data 'YYYYMMDD'
           tile      --tile number of image           
    rtype: list of file links"""
    ## Bands of interest for each sensor type:
    #pdb.set_trace()
    sb = {}
    sb['L30'] = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B10', 'B11', 'Fmask']
    sb['S30'] = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07','B08', 'B8A', 'B11', 'B12', 'Fmask']

    ## TODO: need to figure out how to separate L30 and S30 queries
    sensors = ['L30', 'S30']
    cid = {}
    cid['S30'] = 'collection_concept_id=C2021957295-LPCLOUD'
    cid['L30'] = 'collection_concept_id=C2021957657-LPCLOUD'

    linklist = []
    print('Creating download list')
    for tile in tiles:
        # Request list of file names
        sq = []
        for sensor in sensors:
            sq.append(cid[sensor])
        squeary = '&'.join(sq)
        ## TODO: need to add pagination
        query = "https://cmr.earthdata.nasa.gov/search/granules.json?{}&temporal={}T00:00:00Z,{}T23:59:59Z&attribute[]=string,MGRS_TILE_ID,{}&page_size=2000".format(squeary, sdate, edate, tile)
        ###"https://cmr.earthdata.nasa.gov/search/granules.json?collection_concept_id=C2021957295-LPCLOUD&collection_concept_id=C2021957657-LPCLOUD&temporal=2021-07-01T00:00:00Z,2021-09-01T23:59:59Z&attribute[MGRS_TILE_ID]="+tile
        with requests.get(query, timeout=30) as response:
            if response.status_code == 200:
                res = json.loads(response.text)
                response.close()
                for fil in res['feed']['entry']:
                    #print(fil['links'][0]['href'])
                    for band in fil['links']:
                        if 'title' in band.keys():
                            #print(band['title'])
                            if 'Download' in band['title']:
                                lnk = band['href']
                                snsr = lnk.split('.')[6]
                                b = lnk.split('.')[-2]
                                if b in sb[snsr]:
                                    linklist.append(lnk)
            else:
                print('Bad Query')
                break

    linklist.sort()

    return linklist


def hls_cmask(infile):
    """Function to create cloud mask of HLS image;
    itype: directory of IMG files;
    rtype:"""
    cmask = infile.replace('fmask.tif','cmask_cs.tif')
    if not os.path.exists(cmask):
        print('Creating cloud mask: ', cmask)
        img = gdal.Open(infile)
        gt = img.GetGeoTransform()
        proj = img.GetProjection()
        xsize = img.RasterXSize
        ysize = img.RasterYSize
        nd = img.GetRasterBand(1).GetNoDataValue()
        qa = img.GetRasterBand(1).ReadAsArray()
        qa = np.array(qa, dtype = int)
        # If Cirrus or Cloud or Adjacent Cloud or Cloud Shadow:
        masyk = np.bitwise_and(qa, int('00001111', 2))
        # Nodata -- Do not mask
        qa[np.where(qa == nd)] = 0
        # To mask
        qa[np.where(mask != 0)] = 1
        # Do not mask
        qa[np.where(mask == 0)] = 0
        driver = gdal.GetDriverByName('GTiff')
        output = driver.Create(cmask, xsize, ysize, 1, gdal.GDT_Byte)
        output.SetGeoTransform(gt)
        output.SetProjection(proj)
        output.GetRasterBand(1).WriteArray(qa)
        output = None
        img = None


def download_hls(lks, user, pword, lname):
    """Function downloading HLS Image in batch;
    itype: list of all file links  --[...]
           outdir                  --folder for downloaded image           
    rtype: """

    outdir = r'D:\Imagery\HLSv2.0'
    url = "https://cmr.earthdata.nasa.gov"


    sbands = {'L30': {'B02': 'blue',
                'B03': 'green',
                'B04': 'red',
                'B05': 'nir',
                'B06': 'SWIR_1',
                'B07': 'SWIR_2',
                'B10': 'thermal_infrared_1',
                'B11': 'thermal_infrared_2',
                'Fmask': 'fmask'},
            'S30': {'B02': 'blue',
                'B03': 'green',
                'B04': 'red',
                'B05': 'rededge1',
                'B06': 'rededge2',
                'B07': 'rededge3',
                'B08': 'nir',
                'B8A': 'nir8A', 
                'B11': 'SWIR_1',
                'B12': 'SWIR_2',
                'Fmask': 'fmask'}}

    if lks:
        for lk in lks:
            fname1 = os.path.basename(lk)
            dataset, sensor, tile, doy, v1, v2, band, fmt = fname1.split('.')
            fname2 = fname1.replace(band, sbands[sensor][band])
            fdir = fname1.split(band)[0][:-1]
            stile = tile.lstrip('T')
            year = doy[:4]
            tileFolder = os.path.join(outdir, sensor, year, stile)
            imgFolder = os.path.join(tileFolder, fdir)
            imgFile = os.path.join(imgFolder, fname2)
            # Create data directory if it doesn't already exist
            if not os.path.exists(imgFolder):
                os.makedirs(imgFolder)

            redo = 'yes'
            while redo == 'yes':
                print('Getting link:', lk)
                session = requests.Session()
                session.auth = (user, pword)
                auth = session.post(url)
                #response = session.get(lk, stream=True, timeout=5)
                #with session.get(lk, stream=True) as response:
                success = False
                while not success:
                    try:
                        response = session.get(lk, stream=True, timeout=100)
                        success = True
                    except Exception as e:
                        print('Error! Waiting 60 secs and re-trying...')
                        sys.stdout.flush()
                        time.sleep(60)
                if response.status_code == 200:
                    # Check if file exists/existing file is complete
                    urlsize = response.headers['Content-length']
                    temp = response.raw
                    if os.path.exists(imgFile):
                        exsize = os.path.getsize(imgFile)
                        if urlsize != str(exsize):
                            os.remove(imgFile)
                    # Download only if file does not exist or wrong file size (incomplete download)
                    if not os.path.exists(imgFile):
                        print('Downloading: ', lk)
                        # Keeps timing out
                        with open(imgFile, 'wb') as outputf:
                            shutil.copyfileobj(temp, outputf)
                        #outputf.close()
                        exsize = os.path.getsize(imgFile)
                        if urlsize == str(exsize):
                            redo = 'no'
                            print(urlsize, str(exsize))
                        else:
                            print("Faulty download", urlsize, str(exsize))
                    else: 
                        redo = 'no'
                        print('File previously downloaded')
                response.close()
            if band == 'Fmask':
                hls_cmask(imgFile)
            # append log file with completed download file name
            with open(lname, 'a', newline='') as outf:  
                wobject = writer(outf)
                wobject.writerow([imgFile])
                outf.close()


def main():
    prog = "HLS_download"
    parser = argparse. ArgumentParser(prog=prog, description="HLS download, extract bands, process cloud mask", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-shapefile", type=str, required=True, help="Shapefile for query bounds")
    parser.add_argument("-sdate", type=str, required=True, help="Start date of query. Format yyyy-mm-dd")
    parser.add_argument("-edate", type=str, required=True, help="End date of query. Format yyyy-mm-dd")
    parser.add_argument("-snsr", nargs='+', required=False, default=['L30', 'S30'], help='enter L30 or S30 or L30 S30')
    parser.add_argument("-username", type=str, required=True, help="Earthdata username")
    parser.add_argument("-password", type=str, required=True, help="Earthdata password")
    a = parser.parse_args()
    # Input file and parameters
    shpfl = a.shapefile
    sdate = a.sdate
    edate = a.edate
    user = a.username
    pword = a.password
    sensors = ['L30', 'S30']

    print("Comparing download list to previous log file")
    lname = r"D:\Imagery\HLSv2.0\HLS_v2.0_inventory.txt"

    osbands = {'blueL30': 'B02',
                'greenL30': 'B03',
                'redL30': 'B04',
                'nirL30': 'B05',
                'SWIR_1L30': 'B06',
                'SWIR_2L30': 'B07',
                'thermal_infrared_1L30': 'B10',
                'thermal_infrared_2L30': 'B11',
                'fmaskL30': 'Fmask',
                'blueS30': 'B02',
                'greenS30': 'B03',
                'redS30': 'B04',
                'rededge1S30': 'B05',
                'rededge2S30': 'B06',
                'rededge3S30': 'B07',
                'nirS30': 'B08',
                'nir8AS30': 'B8A', 
                'SWIR_1S30': 'B11',
                'SWIR_2S30': 'B12',
                'fmaskS30': 'Fmask'}

    path_inventory = r"D:\Imagery\HLSv2.0\HLS_v2.0_inventory.txt"
    if os.path.getsize(path_inventory) == 0:
        print("The file is empty.")
        # create an empty dataframe with column names
        lfile = pd.DataFrame(columns=[['bname', 'sensor', 'bname1', 'bname2']])
    else:
        lfile = pd.read_csv(r"D:\Imagery\HLSv2.0\HLS_v2.0_inventory.txt", header=None)
        lfile['bname'] = lfile[0].apply(lambda x: os.path.basename(x))
        lfile['sensor'] = lfile['bname'].apply(lambda x: x.split('.')[1])
        lfile['bname1'] = lfile.apply(lambda x: x['bname'].replace('.tif', x['sensor'] + '.tif'), axis=1)
        lfile['bname2'] = lfile['bname1'].replace(osbands, regex=True)

    # Get tiles based on shapefile input
    dt1 = datetime.now()
    tiles = get_tiles(shpfl)
    dt2 = datetime.now()
    print('Time taken to identify tiles:', dt2-dt1)

    # Get list of files to download
    linklist = list_hls(sensors, sdate, edate, tiles)
    lklst = pd.DataFrame(linklist)
    lklst['bname'] = lklst[0].apply(lambda x: os.path.basename(x))
    if os.path.getsize(path_inventory) != 0:
        lklst.drop(lklst.loc[lklst['bname'].isin(lfile['bname2'].unique().tolist())].index.tolist(), inplace=True)
    linklist = lklst[0].tolist()
    dt3 = datetime.now()
    print('Take taken to identify links:', dt3-dt2)

    # Download files 
    download_hls(linklist, user, pword, lname)
    dt4 = datetime.now()
    print('Time taken to download data:', dt4-dt3)

if __name__ == "__main__":
    main()
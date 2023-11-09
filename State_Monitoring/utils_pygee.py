import ee
import pandas as pd
import os



def ee_array_to_df(arr, list_of_bands):
    """
    Transforms client-side ee.Image.getRegion array to pandas.DataFrame.
    """
    df = pd.DataFrame(arr)

    # Rearrange the header.
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)

    # Remove rows without data inside.
    df = df[['longitude', 'latitude', 'time', *list_of_bands]].dropna()

    # Convert the data to numeric values.
    for band in list_of_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')

    # Convert the time field into a datetime.
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')

    # Keep the columns of interest.
    df = df[['time','datetime',  *list_of_bands]]

    return df



def mask_bits(image_qa, fromBit, toBit):
    def bitwiseExtract(value, fromBit, toBit):
        maskSize = ee.Number(1).add(toBit).subtract(fromBit)
        mask = ee.Number(1).leftShift(maskSize).subtract(1)
        return value.rightShift(fromBit).bitwiseAnd(mask)
    good_qa = bitwiseExtract(image_qa, fromBit, toBit)
    return image_qa.updateMask(good_qa)

def QAfilter(image, QA_band, fromBit, toBit):
    #quality band 
    qaband = image.select(QA_band)
    #good quality image
    good = mask_bits(qaband, fromBit, toBit)
    return image.updateMask(good)
    
def mask_noncrop(image):
    """
    Masks non-crop pixels from the input imag using the MODIS landcover dataset.
    https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD12Q1
    """
    year = image.date().get('year')
    year = ee.Number(image.date().get('year')).max(2001).min(2022) # The landcover dataset is 2001-2022

    landcover_collection = ee.ImageCollection('MODIS/061/MCD12Q1').select('LC_Type1')
    landcover = landcover_collection.filter(ee.Filter.calendarRange(year, year, 'year')).first()
    
    #crop_mask = landcover.gte(12).And(landcover.lte(14))
    crop_mask = landcover.eq(12) 
    return image.updateMask(crop_mask)


def maskClouds_MODIS_Ref(image):
    """
    Function to mask cloudy pixels for MCD43A4
    """
    qa = image.select('Nadir_Reflectance_Band1').select('SummaryQA')
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask)


def maskClouds_MODIS_LAI(image):
    """
    Function to mask cloudy pixels for MCD15A3H
    """
    qa = image.select('Lai_500m').select('FparLai_QC')
    cloudBitMask = 1 << 0
    mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    return image.updateMask(mask)


# Function to calculate NDVI for MCD43A4
def addNDVI_MODIS(image):
    ndvi = image.normalizedDifference(['Nadir_Reflectance_Band2', 'Nadir_Reflectance_Band1']).rename('NDVI')
    return image.addBands(ndvi)
# Function to calculate EVI for MCD43A4
def addEVI_MODIS(image):
    evi = image.expression('2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
        'NIR': image.select('Nadir_Reflectance_Band2'),
        'RED': image.select('Nadir_Reflectance_Band1'),
        'BLUE': image.select('Nadir_Reflectance_Band3')
    }).rename('EVI')
    return image.addBands(evi)

# Extract time series data
def extract_time_series(image, aoi, var):
    image = image.select(var)
    mean_var = image.reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi, scale=500)
    return ee.Feature(None, {'date': image.date().format(), 'mean_{}'.format(var): mean_var.get(var)})

# Convert time series data to pandas dataframe
def eets_to_df(eets_dict, var):
    features = eets_dict['features']
    dates = []
    mean_var = []
    for feature in features:
        properties = feature['properties']
        date = properties.get('date', None)
        mean = properties.get('mean_{}'.format(var), None)
        # Only append to lists if date and mean are not None
        if date:
            dates.append(date)
            mean_var.append(mean if mean is not None else float('NaN'))
    df = pd.DataFrame({'date': dates, 'mean_{}'.format(var): mean_var})
    # Scale the mean_var if var is Lai or Fpar
    if var == 'Lai':
        df['mean_{}'.format(var)] = df['mean_{}'.format(var)] * 0.1
    if var == 'Fpar':
        df['mean_{}'.format(var)] = df['mean_{}'.format(var)] * 0.01
    
    return df

# Aggregate time series data to a longer time interval, e.g. monthly, using pandas 
def agg_ts(df, freq='1M', func = 'median'):
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    if func == 'median':
        df = df.resample(freq).median()
    if func == 'mean':
        df = df.resample(freq).mean()
    if func == 'sum':
        df = df.resample(freq).sum()
    df = df.reset_index()
    return df

# Combine agg_ts and eets_to_df functions
def generate_ts_df(image_collection, aoi, var, freq='1M', func = "median",Output = False, dir_out = None):
    """
    Generate time series data for a given image collection, aoi, and variable.
    """
    # var_ee = ee.String(var)
    # aoi = aoi.geometry()
    
    freq_ee = ee.String(freq)
    eets = image_collection.map(lambda image: extract_time_series(image, aoi, var))
    eets_dict = eets.getInfo()
    df = eets_to_df(eets_dict, var)
    dfagg = agg_ts(df, freq, func = func)
    dfagg['Month'] = dfagg['date'].dt.month
    dfagg['Year'] = dfagg['date'].dt.year

    # Output name 
    aoi_name = aoi.getInfo()['features'][0]['properties']['NAME']
    sdate = df['date'].min().strftime('%Y%m')
    edate = df['date'].max().strftime('%Y%m')
    outname1 = '{}_{}_{}_{}.csv'.format(aoi_name, var, sdate, edate)
    outname2 = '{}_{}_{}_{}_{}.csv'.format(aoi_name, var, sdate, edate, freq)
    if Output:
        df.to_csv(os.path.join(dir_out, outname1), index=False)
        dfagg.to_csv(os.path.join(dir_out, outname2), index=False)
    return df, dfagg

# Function to aggregate time series data on GEE server side
def agg_eets(collection, interval_type, value=30):
    # Calculate the start and end dates of the collection
    start_date = ee.Date(collection.first().get('system:time_start'))
    end_date = ee.Date(collection.sort('system:time_start', False).first().get('system:time_start'))

    if interval_type == "days":
        # Create a list of start times for each interval based on days
        range_list = ee.List.sequence(0, end_date.difference(start_date, 'day'), value)

        def aggregate_by_day_interval(n):
            start = start_date.advance(n, 'day')
            end = start.advance(value, 'day')
            interval_images = collection.filterDate(start, end)
            return interval_images.mean().set('start_date', start).set('end_date', end)

        return range_list.map(aggregate_by_day_interval)

    elif interval_type == "months":
        # Create a list of start months for each interval
        num_months = end_date.difference(start_date, 'month').round()
        range_list = ee.List.sequence(0, num_months.subtract(1))

        def aggregate_by_month(n):
            start = start_date.advance(n, 'month')
            end = start.advance(1, 'month')
            interval_images = collection.filterDate(start, end)
            return interval_images.mean().set('start_date', start).set('end_date', end)

        return range_list.map(aggregate_by_month)

    else:
        raise ValueError("Invalid interval_type. Choose 'days' or 'months'.")
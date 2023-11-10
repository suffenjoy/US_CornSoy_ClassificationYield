# 基于GEE python API提取美国大豆玉米产区监测数据

- 现有数据来源：
  - LAI和FPAR数据来源为MCD15A3H.061 MODIS Leaf Area Index/FPAR 4-Day Global 500m； 
  - NDVI和EVI数据来源为MCD43A4.061 MODIS Nadir BRDF-Adjusted Reflectance Daily 500m。该数据为MODIS反射率数据，通过MODIS反射率数据计算NDVI和EVI，如有需要可以计算其他植被指数；
  - 温度、降雨量和土壤含水量等天气数据来源于CFSV2: NCEP Climate Forecast System Version 2, 6-Hourly Products；
  - 作物掩膜数据来源于MCD12Q1.061 MODIS Land Cover Type Yearly Global 500m;
  - 美国各州边界数据来源于TIGER: US Census States 2018
  
- 所需代码为utils_pygee.py（包含所需要的函数）和State_Monitoring_pygee.ipynb（实际提取时间序列数据）。
- 现有代码支持美国重要的9个玉米大豆生产州：'North Dakota', 'South Dakota', 'Nebraska', 'Kansas', 'Iowa', 'Missouri', 'Illinois', 'Indiana', 'Ohio'。由于各项数据均覆盖全球，因而可以拓展至其他国家和地区，例如巴西和阿根廷等。

- 其他需要设置的变量包括：
  - 提取时间序列的start_date和end_date，由于MODIS植被指数数据为每日数据，CFSV2天气数据为每6小时数据，因而目前代码提取时间序列时不能设置过长的时间跨度，否则会出现云端计算量过大的error。
    - 经小范围测试，提取LAI和FPAR数据可以设置为10年，例如2013-01-01至2023-09-01；
    - 提取NDVI和EVI数据可以设置为3年,例如2020-01-01至2023-09-01；
    - 提取天气数据时需要设置为小于1年，例如2023-01-01至2023-09-01；
  - 将时间序列整合到更大的时间跨度的'freq'，现在默认值是一个月'1M'。
  - 将时间序列整合时所使用的函数，目前所支持的函数有'mean','median','sum'。目前默认设置中，除降雨量为'sum'外，其他数据均为'median'。
  - 数据导出位置。目前会将原时间序列和整合后的时间序列导出到所设置的dir_out中。
    - 原始时间序列的文件名命名方式为："地区名称_变量名称_开始日期_结束日期.csv",例如"Iowa_LAI_20130101_20230901.csv"；
    - 整合后的时间序列的文件名命名方式为："地区名称_变量名称_开始日期_结束日期_整合频率.csv",例如"Iowa_LAI_201301_202309_1M.csv"。


- 未来可以进一步提高的：
  - 其他植被指数的计算和提取
  - 巴西和阿根廷等其他国家和地区的提取
  - 根据实际需求自动化整理所提取的时间序列数据，输出图表等
  - 测试在云端将原始时间序列整合的函数（utils_pygee.py中的agg_eets）以便于提取更长时间跨度的时间序列数据

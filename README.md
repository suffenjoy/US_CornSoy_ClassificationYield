# 基于轮作情况和遥感数据的美国大豆玉米种植面积与产量预测

## 项目简介

本项目旨在进行种植季前（三月前）和种植季早期（六月前）的美国大豆玉米种植面积预测，以及种植季中后期（九月前）的美国大豆玉米产量预测（尚未完成）。
本项目使用的数据主要包括Crop Sequence Boundaries (CSB)数据以及Sentinel-2遥感数据。

## 原数据

- CSB数据下载自网页 https://www.nass.usda.gov/Research_and_Science/Crop-Sequence-Boundaries/index.php 中的2022 Dataset. 该数据包含从2015-2022的8年间基于地块的作物轮作情况，以及各地块的面积和所在地点。下载后的数据包含全美国所有州的地块，体积很大，为geodatabase格式；通过CSB_Subset.py，将其分割为各个州的shapefile格式并导出（目前只导出12个最重要的玉米大豆种植州），之后的数据处理暂时聚焦于伊利诺伊州（Illinois）。
- Sentinel-2遥感数据提取自Google Earth Engine (https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)，是基于CSB地块的时间序列（1月至7月）提取。样本代码（Iowa州，2022）可以通过 https://code.earthengine.google.com/d351221b32c341324ea07f8491bf0cc2 访问，或将ts_GEE.js中的代码复制到Google Earth Engine中运行。
- USDA的验证数据下载自 https://quickstats.nass.usda.gov/

## 经过处理的数据

- GEE中所提取的Sentinel-2时间序列数据: https://drive.google.com/drive/folders/1gFk2kdPyQ9ZDzdn0dmSdFWhCxEt09wnZ?usp=sharing
- 其他数据体积较大，尚未上传完成


## 代码

- CSB_subset.py: 将CSB数据分割为各个州的shapefile格式并导出。
- utils.py: 包含一些常用的函数，如计算植被指数，清理从GEE导出的数据，清理CSB数据等。
- ts_GEE.js: 从GEE提取Sentinel-2 surface reflectance时间序列的代码样本(Iowa州，2022)。
- CSB_based_Mapping_IL.ipynb: 将不同数据集结合，训练模型，预测种植面积。数据基于伊利诺伊州（Illinois）。


## 联系

汤哲寒（Zhehan Tang）
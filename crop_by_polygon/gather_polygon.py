import os
import geopandas as gpd
import mmengine
from osgeo import gdal, osr
from pyproj import CRS

if __name__ == '__main__':
    shp_file = r"E:\polygon\FR_2018\FR_2018_EC21.shp"
    save_folder = 'E:\polygon\generated_files'

    os.makedirs(save_folder, exist_ok=True)

    data = gpd.read_file(shp_file)
    print("原始坐标系:", data.crs)

    imgfile = r"E:\polygon\sentinel-2_region_tiles\T31TEN_20180923T105019_B04_10m.jp2"
    '''
    T31TEN_20180923T105019_B04_10m.jp2 的坐标系是 EPSG:32631
    '''
    input_dataset = gdal.Open(imgfile)
    crs = CRS.from_wkt(input_dataset.GetProjection())

    data = data.to_crs(crs)
    print("转换后坐标系:", data.crs)
    data_list = []
    for idx, row in data.iterrows():
        data_list.append({
            'EC_hcat_c': row['EC_hcat_c'],
            'geometry': row['geometry'],
        })
    data.to_file(save_folder + f'/data_{crs.to_epsg()}.shp')
    mmengine.dump(data_list, save_folder + f'/data_list_{crs.to_epsg()}.pkl')
import os
from functools import partial

import cv2
import numpy as np
import pandas as pd
import ee
import geemap
from matplotlib import pyplot as plt
from osgeo import gdal, osr, ogr
from pyproj import Transformer
import mmengine

ee.Initialize()


def to_proj_4326(src_crs, x, y):
    transformer = Transformer.from_crs(src_crs, 'epsg:4326')
    x, y = transformer.transform(x, y)
    return x, y

def to_4326_osr(x, y, epsg=None, zone_number=None, is_northern_hemisphere=True):
    source_srs = osr.SpatialReference()
    if epsg is not None:
        source_srs.ImportFromEPSG(int(epsg.split(':')[1]))
    else:
        source_srs.SetUTM(zone_number, is_northern_hemisphere)

    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(4326)
    transform = osr.CoordinateTransformation(source_srs, target_srs)
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(x, y)
    point.Transform(transform)

    return point.GetX(), point.GetY()

def download_tif_by_productid(data, vis=False):
    collections = ee.ImageCollection('COPERNICUS/S2_SR')
    product_id = data['S2_PRODUCT_ID']
    geometry = data['geometry']
    img_collection = collections.filter(ee.Filter.eq('PRODUCT_ID', product_id))
    assert img_collection.size().getInfo() == 1
    img = img_collection.first()
    img = img.select('B2', 'B3', 'B4')

    geometry_utm = geometry.split('((')[1].split('))')[0]
    geometry_utm = geometry_utm.replace(', ', ',').replace(' ', ',')
    geometry_utm = geometry_utm.split(',')
    geometry_utm = [[float(geometry_utm[i]), float(geometry_utm[i + 1])] for i in range(0, len(geometry_utm), 2)]
    geometry = [to_proj_4326(data['crs'], *x) for x in geometry_utm]
    geometry = [[x[1], x[0]] for x in geometry]
    mean_lon = np.mean([x[0] for x in geometry[0:4]])
    mean_lat = np.mean([x[1] for x in geometry[0:4]])
    assert np.abs(mean_lon - data['lon']) < 0.001
    assert np.abs(mean_lat - data['lat']) < 0.001

    zone_number = int(data['utm'].split('N')[0])
    is_northern_hemisphere = True if data['utm'].split('N')[1] == 'N' else False
    geometry_osr = [to_4326_osr(*x, None, zone_number, is_northern_hemisphere) for x in geometry_utm]
    assert np.sum(np.abs(np.array(geometry) - np.array(geometry_osr)) > 0.00001)

    geometry = ee.Geometry.Polygon(geometry)

    img_filename = data['save_folder'] + '/' + data['dw_id']
    geemap.download_ee_image(img, filename=img_filename + '.tif', scale=10, region=geometry)
    if vis:
        # vis rgb bands
        rgb_img = visualize_tif(img_filename + '.tif')
        cv2.imwrite(img_filename + '.png', rgb_img)
        # vis label
        label_filename = data['labeler'].replace('-', '_') + '/' + data['hemisphere'] + '/' + str(data['biome']) + '/' + data['dw_id'] + '.tif'
        rgb_label = visualize_label(label_filename)
        plt.imsave(img_filename + '_label.png', rgb_label)

def visualize_tif(tif_file):
    dataset = gdal.Open(tif_file)
    img_gdal = dataset.ReadAsArray()
    img_gdal = img_gdal.transpose(1, 2, 0)
    img_gdal = img_gdal / 8
    img_gdal[img_gdal > 255] = 255
    img_gdal[img_gdal < 0] = 0
    img_gdal = img_gdal.astype('uint8')
    return img_gdal

def visualize_label(label_filename):
    dataset = gdal.Open(label_filename)
    img_gdal = dataset.ReadAsArray()
    color_map = np.zeros_like(img_gdal)
    for i in range(1, 11):
        color_map[img_gdal == i] = i * 25
    color_map = color_map.astype('uint8')
    colors = plt.cm.ScalarMappable(cmap="viridis").to_rgba(color_map)
    colors = colors[:, :, 0:3]
    return colors



if __name__ == '__main__':
    xlsx_file = '副本v1_dw_tile_metadata_for_public_release1.xlsx'
    save_folder = 'downloaded_tif'
    num_workers = 32
    os.makedirs(save_folder, exist_ok=True)

    data = pd.read_excel(xlsx_file)
    data_list = []
    for i in range(100):
        tmp_data_dict = {}
        row = data.iloc[i].to_dict()
        tmp_data_dict.update(row)
        tmp_data_dict['save_folder'] = save_folder
        data_list.append(tmp_data_dict)
    func = partial(download_tif_by_productid, vis=True)
    # data_list = data_list[97:98]
    if num_workers == 1:
        mmengine.track_progress(func, data_list)
    else:
        mmengine.track_parallel_progress(func, data_list, nproc=num_workers)

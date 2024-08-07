import os.path
from functools import partial
import cv2
import mmengine
import numpy as np
import tqdm
from osgeo import gdal, osr
from pyproj import CRS


def crop_img(imgfile, label_mask, img_polygon_bbox, epsg):
    input_dataset = gdal.Open(imgfile)
    crs = CRS.from_wkt(input_dataset.GetProjection())
    current_epsg = crs.to_epsg()
    if current_epsg != epsg:
        print(f"current_epsg: {current_epsg}, epsg: {epsg}")
        return
    geo_transform = input_dataset.GetGeoTransform()
    x_size = input_dataset.RasterXSize
    y_size = input_dataset.RasterYSize

    img_x0 = np.round((geo_transform[0] - img_polygon_bbox[0]) / 10).astype(np.int32)
    img_y0 = np.round((img_polygon_bbox[1] - geo_transform[3]) / 10).astype(np.int32)
    img_x1 = img_x0 + x_size
    img_y1 = img_y0 + y_size

    label_mask_x0 = 0
    label_mask_y0 = 0
    label_mask_x1 = label_mask.shape[1]
    label_mask_y1 = label_mask.shape[0]

    original_img_x0 = 0
    original_img_y0 = 0
    original_img_x1 = x_size
    original_img_y1 = y_size
    # 求交集
    if img_x0 < 0:
        print("img_x0 < 0, set to 0")
        img_x0 = 0
        original_img_y0 = 0 - img_x0
    if img_y0 < 0:
        print("img_y0 < 0, set to 0")
        img_y0 = 0
        original_img_y0 = 0 - img_y0
    if img_x1 > label_mask_x1:
        print("img_x1 > label_mask_x1, set to label_mask_x1")
        img_x1 = label_mask_x1
        original_img_x1 = x_size - (img_x1 - label_mask_x1)
    if img_y1 > label_mask_y1:
        print("img_y1 > label_mask_y1, set to label_mask_y1")
        img_y1 = label_mask_y1
        original_img_y1 = y_size - (img_y1 - label_mask_y1)

    if img_x1 <= 0 or img_y1 <= 0:
        print("没有交集")
        print(f"img_x0: {img_x0}, img_x1: {img_x1}, img_y0: {img_y0}, img_y1: {img_y1}")
        return

    if img_x0 >= img_x1 or img_y0 >= img_y1:
        raise ValueError(f"img_x0: {img_x0}, img_x1: {img_x1}, img_y0: {img_y0}, img_y1: {img_y1}")

    crop_img = input_dataset.ReadAsArray(original_img_x0, original_img_y0, original_img_x1, original_img_y1)
    crop_mask = label_mask[img_y0:img_y1, img_x0:img_x1]

    # 保存
    crop_img_file = imgfile.replace('.jp2', 'crop.tiff')
    crop_mask_file = imgfile.replace('.jp2', 'crop.png')
    # save crop_img as GeoTiff
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(crop_img_file, crop_img.shape[1], crop_img.shape[0], 1, gdal.GDT_UInt16)
    out_dataset.SetGeoTransform((geo_transform[0] + original_img_x0 * 10, 10, 0, geo_transform[3] - original_img_y0 * 10, 0, -10))
    out_dataset.SetProjection(input_dataset.GetProjection())
    out_dataset.GetRasterBand(1).WriteArray(crop_img)
    out_dataset.FlushCache()
    out_dataset = None
    # save crop_mask as PNG
    cv2.imwrite(crop_mask_file, crop_mask)



if __name__ == '__main__':
    img_folder = r"E:\polygon\sentinel-2_region_tiles"
    label_mask_file = r"E:\polygon\generated_files\label_mask.png"
    img_polygon_bbox_file = r"E:\polygon\generated_files\mask_bbox_lon_lat_int.json"

    label_mask = gdal.Open(label_mask_file).ReadAsArray()
    img_polygon_bbox = mmengine.load(img_polygon_bbox_file)
    img_polygon_bbox = np.array(img_polygon_bbox)
    n_proc = 0

    # epsg=32631
    func = partial(crop_img, label_mask=label_mask, img_polygon_bbox=img_polygon_bbox, epsg=32631)
    img_files = mmengine.list_dir_or_file(img_folder, suffix='jp2', list_dir=False)
    img_files = [img_folder + '/' + x for x in img_files]
    img_files = [x for x in img_files if not os.path.exists(x.replace('.jp2', 'crop.png'))]
    # for test
    # img_files = [img_folder+'/T31TEN_20180923T105019_B04_10m.jp2']

    if n_proc > 1:
        mmengine.track_parallel_progress(func, img_files, nproc=n_proc)
    else:
        mmengine.track_progress(func, img_files)

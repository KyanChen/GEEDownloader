# crop the sentinel-2 image by the polygon

## Description
This script is used to crop the sentinel-2 image by the polygon. The input image is the sentinel-2 image, and the input polygon is the polygon that you want to crop the image. The output is the cropped image.

## Usage

1. gather_polygon.py
```bash
python gather_polygon.py
```
This script is used to gather the polygon from a shapefile. The output is the polygon that you want to crop the image in different formats.

2. get_polygon_bbox.py
```bash
python get_polygon_bbox.py
```
This script is used to get the bounding box of the polygon. The output is the bounding box of all the polygons.

3. get_label_mask.py
```bash
python get_label_mask.py
```
This script is used to get the label mask of the polygon. The output is the label mask of all the polygons.

4. crop_label_mask.py
```bash
python crop_label_mask.py
```
This script is used to crop the label mask by the geotiff image. The output is the cropped label mask and corresponding image.

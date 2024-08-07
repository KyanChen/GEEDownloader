import cv2
import mmengine
import numpy as np
import tqdm


def get_mask_bbox(save_folder):
    polygon_bbox = mmengine.load(save_folder + '/mask_bbox_lon_lat.json')
    polygon_bbox = np.array(polygon_bbox)
    polygon_bbox = polygon_bbox / 10
    img_polygon_bbox = np.round(polygon_bbox, 0).astype(np.int32)
    polygon_bbox = img_polygon_bbox * 10
    mmengine.dump(polygon_bbox, save_folder + '/mask_bbox_lon_lat_int.json', indent=4)
    return img_polygon_bbox

def geometry2imgpolygon(geometry, img_polygon_bbox):
    if geometry.geom_type == 'Polygon':
        geometry = [geometry]
    elif geometry.geom_type == 'MultiPolygon':
        geometry = geometry.geoms
    else:
        raise ValueError('geometry type error')
    img_coordinates_list = []
    for polygon in geometry:
        coordinates = list(polygon.exterior.coords)
        coordinates = np.array(coordinates)
        coordinates = coordinates - img_polygon_bbox[:2] * 10
        img_coordinates = np.round(coordinates / 10, 0).astype(np.int32)
        img_coordinates_list.append(img_coordinates)
    return img_coordinates_list


if __name__ == '__main__':
    polygon_file = r"E:\polygon\generated_files\data_list_32631.pkl"
    save_folder = r"E:\polygon\generated_files"
    mmengine.mkdir_or_exist(save_folder)

    img_polygon_bbox = get_mask_bbox(save_folder)
    w, h = img_polygon_bbox[2] - img_polygon_bbox[0], img_polygon_bbox[1] - img_polygon_bbox[2]

    data_list = mmengine.load(polygon_file)


    category2bboxes = {}
    for x in data_list:
        category = x['EC_hcat_c']
        if category not in category2bboxes:
            category2bboxes[category] = []
        category2bboxes[category].append(x['geometry'])

    idx2category = {}
    for idx, category in enumerate(category2bboxes.keys()):
        idx += 1
        idx2category[idx] = category
    # save as json
    mmengine.dump(idx2category, save_folder + '/idx2category.json', indent=4)

    label_mask = np.zeros((h, w), dtype=np.uint8)
    for idx, category in tqdm.tqdm(idx2category.items()):
        idx += 1
        for geometry in category2bboxes[category]:
            img_polygon_list = geometry2imgpolygon(geometry, img_polygon_bbox)
            cv2.fillPoly(label_mask, img_polygon_list, idx)
    cv2.imwrite(save_folder + '/label_mask.png', label_mask)

import mmengine
import numpy as np
import prettytable


if __name__ == '__main__':
    epsg = 32632
    polygon_file = r"E:\polygon\generated_files\data_list_{}.pkl".format(epsg)
    save_folder = 'E:\polygon\generated_files'
    mmengine.mkdir_or_exist(save_folder)
    data_list = mmengine.load(polygon_file)
    bbox_list = [x['geometry'].bounds for x in data_list]
    # 得到最大最小的经纬度
    np_data = np.array(bbox_list)
    width = np_data[:, 2] - np_data[:, 0]
    height = np_data[:, 3] - np_data[:, 1]
    l_lon, l_lat = np.min(np_data[:, 0]), np.max(np_data[:, 3])
    r_lon, r_lat = np.max(np_data[:, 2]), np.min(np_data[:, 1])

    tab = prettytable.PrettyTable()
    tab.field_names = ['category', 'total number', 'w', 'h', 'wh', 'wr', 'hr', 'whr']
    num_total = len(bbox_list)
    num_w_less = np.sum(width < 10)
    num_h_less = np.sum(height < 10)
    num_wh_less = np.sum((width < 10) & (height < 10))
    tab.add_row(['all', len(bbox_list), num_w_less, num_h_less, num_wh_less, np.round(num_w_less / num_total, 2),
                 np.round(num_h_less / num_total, 2), np.round(num_wh_less / num_total, 2)])

    bbox_lon_lat = np.array([l_lon, l_lat, r_lon, r_lat])
    print(bbox_lon_lat)
    mmengine.dump(bbox_lon_lat, save_folder + f'/mask_bbox_lon_lat_{epsg}.json', indent=4)

    category2bboxes = {}
    for x in data_list:
        category = x['EC_hcat_c']
        if category not in category2bboxes:
            category2bboxes[category] = []
        category2bboxes[category].append(x['geometry'].bounds)
    for category, bboxes in category2bboxes.items():
        np_data = np.array(bboxes)
        width = np_data[:, 2] - np_data[:, 0]
        height = np_data[:, 3] - np_data[:, 1]
        num_total = len(bboxes)
        num_w_less = np.sum(width < 10)
        num_h_less = np.sum(height < 10)
        num_wh_less = np.sum((width < 10) & (height < 10))
        tab.add_row([category, num_total, num_w_less, num_h_less, num_wh_less, np.round(num_w_less / num_total, 2),
                     np.round(num_h_less / num_total, 2), np.round(num_wh_less / num_total, 2)])
    print(tab)
    tab_txt = tab.get_string()
    with open(save_folder + f'/category_instances_info_{epsg}.txt', 'w') as f:
        f.write(tab_txt)



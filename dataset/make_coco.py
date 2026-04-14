import os
import numpy as np
import json
import scipy.io as scio
import argparse

def make_index(jsonData: dict, indexDict: dict):
    """
    use coco dict data as orignial data.
    indexDict: {jsonData's key: [index_key, index_value]}
    """
    result = []
    for name in indexDict:
        data = jsonData[name]
        middle_dict = {}
        for item in data:
            if item[indexDict[name][0]] not in middle_dict:
                middle_dict.update({item[indexDict[name][0]]: [item[indexDict[name][1]]]})
            else:
                middle_dict[item[indexDict[name][0]]].append(item[indexDict[name][1]])
        result.append(middle_dict)

    return result

def check_file_exist(indexDict: dict, file_path: str):
    keys = list(indexDict.keys())
    for item in keys:
        # print(indexDict[item])
        if not os.path.exists(os.path.join(file_path, indexDict[item][0])):
            print(item, indexDict[item])
            indexDict.pop(item)
        indexDict[item] = os.path.join(file_path, indexDict[item][0])
    return indexDict

def chage_categories2numpy(category_ids: dict, data: dict):
    
    for item in data:
        class_item = [0] * len(category_ids)
        for class_id in data[item]:
            class_item[category_ids[class_id]] = 1
        data[item] = np.asarray(class_item)

    return data

def get_all_use_key(categoryDict: dict):
    return list(categoryDict.keys())

def remove_not_use(data: dict, used_key: list):

    keys = list(data.keys())
    for item in keys:
        if item not in used_key:
            # print("remove:", item, indexDict[item])
            data.pop(item)
    # print(len(category_list))
    return data

def merge_to_list(data: dict):

    result = []
    key_sort = list(data.keys())
    key_sort.sort()
    # print(key_sort)
    # print(key_sort.index(91654))

    for item in key_sort:
        result.append(data[item])

    return result


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-dir", default="/remote-home/zhangli/20250601_DSCPH/guyue_model/DScPH-main/dataset/coco2017/coco2017", type=str, help="the coco dataset dir")
    parser.add_argument("--save-dir", default="/remote-home/zhangli/20250601_DSCPH/guyue_model/DScPH-main/dataset/coco", type=str, help="mat file saved dir")
    args = parser.parse_args()
    
    PATH = args.coco_dir
    
    print("Processing training set...")
    
    jsonFile = os.path.join(PATH, "annotations", "captions_train2017.json")
    with open(jsonFile, "r") as f:
        jsonData = json.load(f)
    indexDict = {"images": ["id", "file_name"], "annotations": ["image_id", "caption"]}
    result = make_index(jsonData, indexDict)
    indexDict_, captionDict = result
    indexDict_ = check_file_exist(indexDict_, os.path.join(PATH, "train2017"))
    print("caption dict sizes:", len(indexDict_), len(captionDict))
    
    jsonFile = os.path.join(PATH, "annotations", "instances_train2017.json")
    with open(jsonFile, "r") as f:
        jsonData = json.load(f)
    categroy_ids = {}
    for i, item in enumerate(jsonData['categories']):
        categroy_ids.update({item['id']: i})
    indexDict = {"annotations": ["image_id", "category_id"], "images": ["id", "file_name"]}
    result = make_index(jsonData, indexDict)
    categoryDict = result[0]
    cateIndexDict = result[1]
    categoryDict = chage_categories2numpy(categroy_ids, categoryDict)

    all_keys_set = set(indexDict_.keys())
    all_keys_set = all_keys_set & set(captionDict.keys())
    all_keys_set = all_keys_set & set(categoryDict.keys())
    all_keys_set = all_keys_set & set(cateIndexDict.keys())
    common_keys = sorted(list(all_keys_set))
    
    print(f"Training set - Common keys: {len(common_keys)}")

    def filter_with_common_keys(data_dict, common_keys):
        return [data_dict[k] for k in common_keys]
    
    indexList_train = filter_with_common_keys(indexDict_, common_keys)
    captionList_train = filter_with_common_keys(captionDict, common_keys)
    categoryList_train = filter_with_common_keys(categoryDict, common_keys)
    categoryIndexList_train = filter_with_common_keys(cateIndexDict, common_keys)

    assert len(indexList_train) == len(captionList_train) == len(categoryList_train) == len(categoryIndexList_train), \
        "Training set lists must be of the same length after alignment!"
    print(f"Training set aligned with length: {len(indexList_train)}")

    print("Processing validation set...")

    val_jsonFile = os.path.join(PATH, "annotations", "captions_val2017.json")
    with open(val_jsonFile, "r") as f:
         jsonData = json.load(f)
    indexDict = {"images": ["id", "file_name"], "annotations": ["image_id", "caption"]}
    result = make_index(jsonData, indexDict)
    val_indexDict = result[0]
    val_captionDict = result[1]
    val_indexDict = check_file_exist(val_indexDict, os.path.join(PATH, "val2017"))
  
    jsonFile = os.path.join(PATH, "annotations", "instances_val2017.json")
    with open(jsonFile, "r") as f:
        jsonData = json.load(f)
    categroy_ids = {}
    for i, item in enumerate(jsonData['categories']):
        categroy_ids.update({item['id']: i})
    indexDict = {"annotations": ["image_id", "category_id"], "images": ["id", "file_name"]}
    result = make_index(jsonData, indexDict)
    val_categoryDict = result[0]
    val_categoryIndexDict = result[1]
    val_categoryDict = chage_categories2numpy(categroy_ids, val_categoryDict)
   
    val_all_keys_set = set(val_indexDict.keys())
    val_all_keys_set = val_all_keys_set & set(val_captionDict.keys())
    val_all_keys_set = val_all_keys_set & set(val_categoryDict.keys())
    val_all_keys_set = val_all_keys_set & set(val_categoryIndexDict.keys())
    val_common_keys = sorted(list(val_all_keys_set))
    
    print(f"Validation set - Common keys: {len(val_common_keys)}")
   
    val_indexList = filter_with_common_keys(val_indexDict, val_common_keys)
    val_captionList = filter_with_common_keys(val_captionDict, val_common_keys)
    val_categoryIndexList = filter_with_common_keys(val_categoryIndexDict, val_common_keys)
    val_categoryList = filter_with_common_keys(val_categoryDict, val_common_keys)

    assert len(val_indexList) == len(val_captionList) == len(val_categoryList) == len(val_categoryIndexList), \
        "Validation set lists must be of the same length after alignment!"
    print(f"Validation set aligned with length: {len(val_indexList)}")

    indexList = indexList_train + val_indexList
    captionList = captionList_train + val_captionList
    categoryIndexList = categoryIndexList_train + val_categoryIndexList
    categoryList = categoryList_train + val_categoryList

    print(f"Final dataset sizes - Index: {len(indexList)}, Caption: {len(captionList)}, Category: {len(categoryList)}")

    assert len(indexList) == len(captionList) == len(categoryList), \
        "Final lists must be of the same length!"

    indexs = {"index": indexList}
    captions = {"caption": captionList}
    categorys = {"category": categoryList}

    os.makedirs(args.save_dir, exist_ok=True)
    
    scio.savemat(os.path.join(args.save_dir, "index.mat"), indexs)
    scio.savemat(os.path.join(args.save_dir, "caption.mat"), captions)
    scio.savemat(os.path.join(args.save_dir, "label.mat"), categorys)
    
    print(f"Data successfully saved to {args.save_dir}")
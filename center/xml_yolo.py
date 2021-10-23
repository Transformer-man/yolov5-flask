import os
from tqdm import tqdm
from lxml import etree
import json
import shutil
# 原始xml路径和image路径
xml_root_path = r'../datasets/VOC2012/Annotations'
img_root_path = r'../datasets/VOC2012/JPEGImages'
# 保存的图片和yolo格式label路径。要新建文件夹
def get_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path
get_path(r'../datasets/yolo_data')
save_label_path = get_path(r'../datasets/yolo_data/labels')
save_images_path = get_path(r'../datasets/yolo_data/images')
def parse_xml_to_dict(xml):
    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}
def translate_info(file_names, img_root_path, class_list):
    for root,dirs,files in os.walk(file_names):
        for file in tqdm(files):
            # 检查xml文件是否存在
            xml_path = os.path.join(root, file)
            # read xml
            with open(xml_path) as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = parse_xml_to_dict(xml)["annotation"]
            img_height = int(data["size"]["height"])
            img_width = int(data["size"]["width"])
            img_path = data["filename"]

            # write object info into txt
            assert "object" in data.keys(), "file: '{}' lack of object key.".format(xml_path)
            if len(data["object"]) == 0:
                # 如果xml文件中没有目标就直接忽略该样本
                print("Warning: in '{}' xml, there are no objects.".format(xml_path))
                continue
            with open(os.path.join(save_label_path, file.split(".")[0] + ".txt"), "w") as f:
                for index, obj in enumerate(data["object"]):
                    # 获取每个object的box信息
                    xmin = float(obj["bndbox"]["xmin"])
                    xmax = float(obj["bndbox"]["xmax"])
                    ymin = float(obj["bndbox"]["ymin"])
                    ymax = float(obj["bndbox"]["ymax"])
                    class_name = obj["name"]
                    class_index = class_list.index(class_name)
                    # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
                    if xmax <= xmin or ymax <= ymin:
                        print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                        continue
                    # 将box信息转换到yolo格式
                    xcenter = xmin + (xmax - xmin) / 2
                    ycenter = ymin + (ymax - ymin) / 2
                    w = xmax - xmin
                    h = ymax - ymin
                    # 绝对坐标转相对坐标，保存6位小数
                    xcenter = round(xcenter / img_width, 6)
                    ycenter = round(ycenter / img_height, 6)
                    w = round(w / img_width, 6)
                    h = round(h / img_height, 6)
                    info = [str(i) for i in [class_index, xcenter, ycenter, w, h]]
                    if index == 0:
                        f.write(" ".join(info))
                    else:
                        f.write("\n" + " ".join(info))
            # copy image into save_images_path
            path_copy_to = os.path.join(save_images_path,file.split(".")[0] + ".jpg")
            shutil.copyfile(os.path.join(img_root_path, img_path), path_copy_to)
            
label_json_path = r'../datasets/VOC2012/pascal_voc_classes.txt'
with open(label_json_path, 'r') as f:
    label_file = f.readlines()
class_list = label_file[0].split(',')
translate_info(xml_root_path, img_root_path, class_list)
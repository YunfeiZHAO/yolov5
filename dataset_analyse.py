# This script take the helper function in cityscape as a reference
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/annotation.py
# I try to extend it to a uniform dataset processing script

import os
import json
import cv2
import random
import re
from bs4 import BeautifulSoup as bs
from shutil import copyfile
from tqdm import tqdm

from abc import ABCMeta, abstractmethod

from pathlib import Path


class CsObjectType():
    """Type of an object"""
    POLY = 1  # polygon
    BBOX2D = 2  # bounding box
    BBOX3D = 3  # 3d bounding box
    IGNORE2D = 4  # 2d ignore region


class CsObject:
    """Abstract base class for annotation objects"""
    __metaclass__ = ABCMeta

    def __init__(self, obj_type):
        self.objectType = obj_type
        # the label
        self.label = ""

    @abstractmethod
    def __str__(self): pass

    @abstractmethod
    def json_to_txt(self): pass


class IDDBbox2d(CsObject):
    """
    Class that contains the information of a single annotated object as bounding box
    For dataset https://idd.insaan.iiit.ac.in/dataset/download/
    """

    def __init__(self):
        CsObject.__init__(self, CsObjectType.BBOX2D)
        # the polygon as list of points
        self.bbox_xyxy = []
        # the label of the corresponding object
        self.label = ""

    def __str__(self):
        bboxText = ""
        bboxText += '[(x_min: {}, y_min: {}), (x_max: {}, y_max: {})]'.format(
            self.bbox_xyxy[0], self.bbox_xyxy[1], self.bbox_xyxy[2], self.bbox_xyxy[3])

        text = "Object: {}\n -  Bbox {}".format(
            self.label, bboxText)
        return text

    # access 2d boxes in [xmin, ymin, xmax, ymax] format

    @property
    def bbox_xywh(self):
        """Returns the 2d box as [x_center, y_center, w,h]"""
        return [
            (self.bbox_xyxy[0] + self.bbox_xyxy[2]) // 2,
            (self.bbox_xyxy[1] + self.bbox_xyxy[3]) // 2,
            self.bbox_xyxy[2] - self.bbox_xyxy[0],
            self.bbox_xyxy[3] - self.bbox_xyxy[1]
        ]

    def fromXMLText(self, object):
        """
         try to load from a  bs4.element.Tag
         object: bs4.element.Tag extracted from xml file
        """
        self.label = object.find('name').contents[0]
        if self.label is None:
            print('box without label')

        self.bbox_xyxy.append(int(object.find('xmin').contents[0]))
        self.bbox_xyxy.append(int(object.find('ymin').contents[0]))
        self.bbox_xyxy.append(int(object.find('xmax').contents[0]))
        self.bbox_xyxy.append(int(object.find('ymax').contents[0]))

    def toJsonText(self):
        objDict = {}
        objDict['label'] = self.label
        objDict['bbox'] = self.bbox_xyxy
        return objDict


class IDDAnnotation:
    """The annotation of a whole image (doesn't support mixed annotations, i.e. combining CsPoly and CsBbox2d)
        Extract annotation from XML file
    """

    # Constructor
    def __init__(self, objType=CsObjectType.BBOX2D):
        # the width of that image and thus of the label image
        self.imgWidth = 0
        # the height of that image and thus of the label image
        self.imgHeight = 0
        # the list of objects
        self.objects = []
        # the camera calibration
        self.camera = None
        assert objType in CsObjectType.__dict__.values()
        self.objectType = objType

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def from_xml_txt(self, xmlText):
        bs_data = bs(xmlText, 'xml')
        self.imgWidth = int(bs_data.find('width').contents[0])
        self.imgHeight = int(bs_data.find('height').contents[0])
        self.objects = []
        # load objects
        for object in bs_data.findAll('object'):
            if self.objectType == CsObjectType.BBOX2D:
                obj = IDDBbox2d()
                obj.fromXMLText(object)
                self.objects.append(obj)

    def toJsonText(self):
        jsonDict = {}
        jsonDict['imgWidth'] = self.imgWidth
        jsonDict['imgHeight'] = self.imgHeight
        jsonDict['objects'] = []
        for obj in self.objects:
            objDict = obj.toJsonText()
            jsonDict['objects'].append(objDict)
        return jsonDict

    # Read a xml formatted file and return the annotation
    def from_xml_file(self, xml_file):
        if not os.path.isfile(xml_file):
            print('Given XML file not found: {}'.format(xml_file))
            return
        with open(xml_file, 'r') as f:
            xml_text = f.read()
            self.from_xml_txt(xml_text)

    def toJsonFile(self, jsonFile):
        with open(jsonFile, 'w') as f:
            f.write(self.toJson())


def get_file_paths(path='./gtBboxCityPersons/train/'):
    """Get all file paths from a parent path"""
    for dir_path, dir_names, files in os.walk(path):
        for file in files:
            yield os.path.join(dir_path, file)


def count_files(path='./gtBboxCityPersons/train/'):
    """count files from parent path"""
    f = get_file_paths(path)
    print(f'Number of files in {path}: {len(list(f))}')


def IDD_label_count(annotation_folder):
    """count number of labels"""
    labels = {}
    annotations = get_file_paths(annotation_folder)
    for f in annotations:
        print(f)
        if f.split('.')[-1] == 'xml':
            annotation = IDDAnnotation()
            annotation.from_xml_file(f)
            for obj in annotation.objects:
                if obj.label in labels:
                    labels[obj.label] += 1
                else:
                    labels[obj.label] = 1
    print(labels)


def IDD_mean_number_object(annotation_folder):
    """count number of labels"""
    n_object = 0
    n_annotation = 0
    annotations = get_file_paths(annotation_folder)
    for f in annotations:
        if f.split('.')[-1] == 'xml':
            annotation = IDDAnnotation()
            annotation.from_xml_file(f)
            n_object += len(annotation.objects)
            n_annotation += 1
    print(f'mean object in image: ', n_object / n_annotation)


def txt_file_element_count(path='/home/yunfei/Desktop/IDD_Detection/train.txt'):
    with open(path, 'r') as f:
        xml_text = f.read()
        a = xml_text.split('\n')
        print('Number of element:', (len(a) - 1))


def IDD_json_annotation_generation(root_path='../IDD_Detection', type='train',
                                   jsonfile_root='../IDD_Detection/Annotations'):
    annotation_path = os.path.join(root_path, 'Annotations')
    image_path = os.path.join('JPEGImages')
    txt_file_path = os.path.join(root_path, f'{type}.txt')
    dataset = {'info': {'name': 'IDD dataset',
                        'website': 'https://idd.insaan.iiit.ac.in/dataset/details/'}}

    categories = {'vehicle fallback': 1, 'rider': 2, 'bus': 3, 'car': 4, 'autorickshaw': 5, 'truck': 6,
                  'motorcycle': 7, 'person': 8, 'traffic sign': 9, 'animal': 10, 'bicycle': 11,
                  'traffic light': 12, 'caravan': 13, 'train': 14, 'trailer': 15}
    cats = []
    for cat, ind in categories.items():
        cats.append({'id': ind, 'name': cat})
    dataset['categories'] = cats

    images = []
    annotations = []
    image_id = 0
    annotation_id = 0
    with open(txt_file_path) as fp:
        for line in tqdm(fp):
            item_path = line.strip()
            annotation_file_path = os.path.join(annotation_path, f'{item_path}.xml')
            image_file_path = os.path.join(image_path, f'{item_path}.jpg')
            image_annotation = IDDAnnotation()
            image_annotation.from_xml_file(annotation_file_path)

            for object in image_annotation.objects:
                annotation = {'id': annotation_id,
                              'image_id': image_id,
                              'category_id': categories[object.label],
                              'bbox': object.bbox_xyxy}
                annotations.append(annotation)
                annotation_id += 1

            image = {'file_name': image_file_path,
                     'height': image_annotation.imgHeight,
                     'width': image_annotation.imgWidth,
                     'id': image_id}
            images.append(image)
            image_id += 1

    dataset['annotations'] = annotations
    dataset['images'] = images
    with open(os.path.join(jsonfile_root, f'{type}.json'), 'w') as outfile:
        json.dump(dataset, outfile)
    return


def IDD_to_COCO(root_path='../IDD_yolo', type='train'):
    """refactoration for IDD dataset to coco dataset format"""

    annotation_path = os.path.join(root_path, 'Annotations')
    image_path = os.path.join(root_path, 'JPEGImages')
    txt_file_path = os.path.join(root_path, f'{type}.txt')
    categories = {'vehicle fallback': 0, 'rider': 1, 'bus': 2, 'car': 3, 'autorickshaw': 4, 'truck': 5,
                  'motorcycle': 6, 'person': 7, 'traffic sign': 8, 'animal': 9, 'bicycle': 10,
                  'traffic light': 11, 'caravan': 12, 'train': 13, 'trailer': 14}
    im_folder = os.path.join(root_path, 'images', type)
    label_folder = os.path.join(root_path, 'labels', type)
    Path(im_folder).mkdir(parents=True, exist_ok=True)
    Path(label_folder).mkdir(parents=True, exist_ok=True)
    with open(txt_file_path, 'r') as fp:
        for line in tqdm(fp):
            item_path = line.strip()
            annotation_file_path = os.path.join(annotation_path, f'{item_path}.xml')
            image_file_path = os.path.join(image_path, f'{item_path}.jpg')
            new_txt_label_name = f"{item_path.split('/')[-1]}.txt"
            new_txt_label_path = os.path.join(label_folder, new_txt_label_name)

            image_annotation = IDDAnnotation()
            image_annotation.from_xml_file(annotation_file_path)

            imgWidth = image_annotation.imgWidth
            imgHeight = image_annotation.imgHeight
            with open(new_txt_label_path, 'w') as new_annotation_file:
                for bbox_object in image_annotation.objects:
                    class_number = categories[bbox_object.label]
                    x, y, xm, ym = bbox_object.bbox_xyxy
                    x_center = ((x + xm)/2)/imgWidth
                    y_center = ((y + ym)/2)/imgHeight
                    w = (xm - x)/imgWidth
                    h = (ym - y)/imgHeight
                    line = f'{class_number} {x_center} {y_center} {w} {h}\n'
                    new_annotation_file.write(line)
            #os.system(f"cp {image_file_path} {im_folder}")

    # create new txt file
    origin_txt_path = os.path.join(root_path, 'origin_txt')
    Path(origin_txt_path).mkdir(parents=True, exist_ok=True)
    with open(txt_file_path, 'r') as fp:
        lines = fp.readlines()
    os.system(f"mv {txt_file_path} {origin_txt_path}")

    with open(txt_file_path, 'w') as fp:
        for line in lines:
            item_path = line.strip()
            imaga_name = f"{item_path.split('/')[-1]}.jpg"
            new_image_path = f"./images/{type}/{imaga_name}\n"
            fp.write(new_image_path)






def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def show_image_annotated_txt(annotation_path, image_path):
    """Need annotation txt file and image file path"""
    img = cv2.imread(image_path)
    im_h, im_w, _ = img.shape
    with open(annotation_path, 'r') as fp:
        for line in fp:
            label, x, y, w, h = line.split()
            x, y, w, h = float(x), float(y), float(w), float(h)
            x, w = x*im_w, w*im_w
            y, h = y*im_h, h*im_h
            plot_one_box((int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)), img, label=label, line_thickness=3)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# def generate_yolo_label_files(cityscapes_annotation_folder='./gtBboxCityPersons/train/', yolo_annotation_folder='.', name='train'):
#     """generate label txt file, to be noted, the label prefix should be the same as image prefix
#     :param cityscapes_annotation_folder: folder where we store labels of cityscapes
#     :param yolo_annotation_folder: parent folder to create yolo labels
#     :param name: name of the label folder  eg. train, val
#     """
#     files = sorted(list(get_file_paths(path=cityscapes_annotation_folder)))
#     save_folder = os.path.join(yolo_annotation_folder, 'labels', name)
#     os.makedirs(save_folder, exist_ok=True)
#     for f in files:
#         annotation = Annotation()
#         annotation.fromJsonFile(f)
#         imgWidth = annotation.imgWidth
#         imgHeight = annotation.imgHeight
#         filename = re.split('/|\.', f)[-2]  # split by / and .
#         # change file name to make it the same as images
#         filename = re.sub('gtBboxCityPersons', 'leftImg8bit', filename)
#         save_path = os.path.join(save_folder, f'{filename}.txt')
#         save_file = open(save_path, "w")
#         for obj in annotation.objects:
#             x, y, xm, ym = obj.bbox_modal
#             x_center = ((x + xm)/2)/imgWidth
#             y_center = ((y + ym)/2)/imgHeight
#             w = (xm - x)/imgWidth
#             h = (ym - y)/imgHeight
#             class_number = cityscape_label_dict[obj.label]
#             line = f'{class_number} {x_center} {y_center} {w} {h}\n'
#             save_file.write(line)
#         save_file.close()
#
#
# def generate_yolo_image_path_file(cityscapes_image_folder='./leftImg8bit/train/', yolo_image_file='./train.txt', image_path='.s/images/train'):
#     """generate image path txt file
#     :param cityscapes_image_folder: folder where we store images of cityscapes
#     :param yolo_image_file: parent folder to create yolo image path file
#     :param image_path: the folder we store image  eg. ./images/train, ./images/val
#     """
#     files = sorted(list(get_file_paths(path=cityscapes_image_folder)))
#     save_file = open(yolo_image_file, 'w')
#     for f in files:
#         filename = re.split('/|\.', f)[-2]
#         file_path = os.path.join(image_path, f'{filename}.png')
#         save_file.write(f'{file_path}\n')
#     save_file.close()
#
#
# def create_image_folder(cityscapes_image_folder='./leftImg8bit/train/', image_path='./images/train'):
#     """ Create image folder which can be used by yolov5
#     :param cityscapes_image_folder: the folder where cityscape store images
#     :param image_path: the place to store images
#     """
#     files = sorted(list(get_file_paths(path=cityscapes_image_folder)))
#     os.makedirs(image_path, exist_ok=True)
#     for f in files:
#         filename = re.split('/', f)[-1]
#         copyfile(f, os.path.join(image_path, filename))

# IDD labels
# {'vehicle fallback': 21081, 'rider': 97626, 'bus': 18745, 'car': 90520, 'autorickshaw': 32280, 'truck': 27837, 'motorcycle': 103608, 'person': 88397, 'traffic sign': 14203, 'animal': 6224, 'bicycle': 3142, 'traffic light': 3699, 'caravan': 136, 'train': 60, 'trailer': 18}


def main():
    # count_files(path='/home/yunfei/Desktop/IDD_Detection/Annotations')
    # count_files(path='/home/yunfei/Desktop/IDD_Detection/JPEGImages')
    # txt_file_element_count('/home/yunfei/Desktop/IDD_Detection/train.txt')
    # txt_file_element_count('/home/yunfei/Desktop/IDD_Detection/val.txt')
    # txt_file_element_count('/home/yunfei/Desktop/IDD_Detection/test.txt')
    # IDD_label_count('/home/yunfei/Desktop/IDD_Detection/Annotations')
    # IDD_mean_number_object('/home/yunfei/Desktop/IDD_Detection/Annotations')
    # IDD_json_annotation_generation(root_path='../IDD_Detection', type='train',
    #                                jsonfile_root='../IDD_Detection/Annotations')
    # IDD_json_annotation_generation(root_path='../IDD_Detection', type='val',
    #                               jsonfile_root='../IDD_Detection/Annotations')
    # IDD_json_annotation_generation(root_path='../IDD_Detection', type='test',
    #                                jsonfile_root='../IDD_Detection/Annotations')
    #
    IDD_to_COCO(root_path='../IDD_yolo', type='val')
    IDD_to_COCO(root_path='../IDD_yolo', type='train')

    # show_image_annotated_txt('/home/yunfei/Desktop/IDD_yolo/labels/train/001542_r.txt', '/home/yunfei/Desktop/IDD_yolo/images/train/001542_r.jpg')

if __name__ == '__main__':
    main()

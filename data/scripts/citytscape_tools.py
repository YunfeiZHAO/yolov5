# This script take the helper function in cityscape as a reference
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/annotation.py

import os
import json
import cv2
import random
import re
from shutil import copyfile

from abc import ABCMeta, abstractmethod


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


class CsBbox2d(CsObject):
    """Class that contains the information of a single annotated object as bounding box"""
    def __init__(self):
        CsObject.__init__(self, CsObjectType.BBOX2D)
        # the polygon as list of points
        self.bbox_amodal_xywh = []
        self.bbox_modal_xywh = []
        # the ID of the corresponding object
        self.instanceId = -1
        # the label of the corresponding object
        self.label = ""

    def __str__(self):
        bboxAmodalText = ""
        bboxAmodalText += '[(x1: {}, y1: {}), (w: {}, h: {})]'.format(
            self.bbox_amodal_xywh[0], self.bbox_amodal_xywh[1],  self.bbox_amodal_xywh[2],  self.bbox_amodal_xywh[3])

        bboxModalText = ""
        bboxModalText += '[(x1: {}, y1: {}), (w: {}, h: {})]'.format(
            self.bbox_modal_xywh[0], self.bbox_modal_xywh[1], self.bbox_modal_xywh[2], self.bbox_modal_xywh[3])

        text = "Object: {}\n - Amodal {}\n - Modal {}".format(
            self.label, bboxAmodalText, bboxModalText)
        return text

    # access 2d boxes in [xmin, ymin, xmax, ymax] format
    @property
    def bbox_amodal(self):
        """Returns the 2d box as [xmin, ymin, xmax, ymax]"""
        return [
            self.bbox_amodal_xywh[0],
            self.bbox_amodal_xywh[1],
            self.bbox_amodal_xywh[0] + self.bbox_amodal_xywh[2],
            self.bbox_amodal_xywh[1] + self.bbox_amodal_xywh[3]
        ]

    @property
    def bbox_modal(self):
        """Returns the 2d box as [xmin, ymin, xmax, ymax]"""
        return [
            self.bbox_modal_xywh[0],
            self.bbox_modal_xywh[1],
            self.bbox_modal_xywh[0] + self.bbox_modal_xywh[2],
            self.bbox_modal_xywh[1] + self.bbox_modal_xywh[3]
        ]

    def fromJsonText(self, jsonText, objId=-1):
        # try to load from cityperson format
        if 'bbox' in jsonText.keys() and 'bboxVis' in jsonText.keys():
            self.bbox_amodal_xywh = jsonText['bbox']
            self.bbox_modal_xywh = jsonText['bboxVis']
        # both modal and amodal boxes are provided
        elif "modal" in jsonText.keys() and "amodal" in jsonText.keys():
            self.bbox_amodal_xywh = jsonText['amodal']
            self.bbox_modal_xywh = jsonText['modal']
        # only amodal boxes are provided
        else:
            self.bbox_modal_xywh = jsonText['amodal']
            self.bbox_amodal_xywh = jsonText['amodal']

        # load label and instanceId if available
        if 'label' in jsonText.keys() and 'instanceId' in jsonText.keys():
            self.label = str(jsonText['label'])
            self.instanceId = jsonText['instanceId']

    def toJsonText(self):
        objDict = {}
        objDict['label'] = self.label
        objDict['instanceId'] = self.instanceId
        objDict['modal'] = self.bbox_modal_xywh
        objDict['amodal'] = self.bbox_amodal_xywh

        return objDict


class Annotation:
    """The annotation of a whole image (doesn't support mixed annotations, i.e. combining CsPoly and CsBbox2d)"""

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

    def fromJsonText(self, jsonText):
        jsonDict = json.loads(jsonText)
        self.imgWidth = int(jsonDict['imgWidth'])
        self.imgHeight = int(jsonDict['imgHeight'])
        self.objects = []
        # load objects
        if self.objectType != CsObjectType.IGNORE2D:
            for objId, objIn in enumerate(jsonDict['objects']):
                if self.objectType == CsObjectType.BBOX2D:
                    obj = CsBbox2d()
                obj.fromJsonText(objIn, objId)
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

    # Read a json formatted polygon file and return the annotation
    def fromJsonFile(self, jsonFile):
        if not os.path.isfile(jsonFile):
            print('Given json file not found: {}'.format(jsonFile))
            return
        with open(jsonFile, 'r') as f:
            jsonText = f.read()
            self.fromJsonText(jsonText)

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


def show_image_annotated(annotation_path, image_path):
    """Need annotation json file and image file path"""
    annotation = Annotation()
    annotation.fromJsonFile(annotation_path)
    img = cv2.imread(image_path)
    for obj in annotation.objects:
        xyxy = obj.bbox_modal
        plot_one_box(xyxy, img, label=obj.label, line_thickness=3)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def label_count(annotation_folder):
    """count number of labelsnhhhhhnn"""
    labels = {}
    annotations = get_file_paths(annotation_folder)
    for f in annotations:
        annotation = Annotation()
        annotation.fromJsonFile(f)
        for obj in annotation.objects:
            if obj.label in labels:
                labels[obj.label] += 1
            else:
                labels[obj.label] = 1
    print(labels)


label_dict = {
    'pedestrian':0,
    'rider':1,
    'sitting person':2,
    'person (other)':3,
    'person group':4,
    'ignore':5
}


def generate_yolo_label_files(cityscapes_annotation_folder='./gtBboxCityPersons/train/', yolo_annotation_folder='.', name='train'):
    """generate label txt file, to be noted, the label prefix should be the same as image prefix
    :param cityscapes_annotation_folder: folder where we store labels of cityscapes
    :param yolo_annotation_folder: parent folder to create yolo labels
    :param name: name of the label folder  eg. train, val
    """
    files = sorted(list(get_file_paths(path=cityscapes_annotation_folder)))
    save_folder = os.path.join(yolo_annotation_folder, 'labels', name)
    os.makedirs(save_folder, exist_ok=True)
    for f in files:
        annotation = Annotation()
        annotation.fromJsonFile(f)
        imgWidth = annotation.imgWidth
        imgHeight = annotation.imgHeight
        filename = re.split('/|\.', f)[-2]  # split by / and .
        # change file name to make it the same as images
        filename = re.sub('gtBboxCityPersons', 'leftImg8bit', filename)
        save_path = os.path.join(save_folder, f'{filename}.txt')
        save_file = open(save_path, "w")
        for obj in annotation.objects:
            x, y, xm, ym = obj.bbox_modal
            x_center = ((x + xm)/2)/imgWidth
            y_center = ((y + ym)/2)/imgHeight
            w = (xm - x)/imgWidth
            h = (ym - y)/imgHeight
            class_number = label_dict[obj.label]
            line = f'{class_number} {x_center} {y_center} {w} {h}\n'
            save_file.write(line)
        save_file.close()


def generate_yolo_image_path_file(cityscapes_image_folder='./leftImg8bit/train/', yolo_image_file='./train.txt', image_path='.s/images/train'):
    """generate image path txt file
    :param cityscapes_image_folder: folder where we store images of cityscapes
    :param yolo_image_file: parent folder to create yolo image path file
    :param image_path: the folder we store image  eg. ./images/train, ./images/val
    """
    files = sorted(list(get_file_paths(path=cityscapes_image_folder)))
    save_file = open(yolo_image_file, 'w')
    for f in files:
        filename = re.split('/|\.', f)[-2]
        file_path = os.path.join(image_path, f'{filename}.png')
        save_file.write(f'{file_path}\n')
    save_file.close()


def create_image_folder(cityscapes_image_folder='./leftImg8bit/train/', image_path='./images/train'):
    """ Create image folder which can be used by yolov5
    :param cityscapes_image_folder: the folder where cityscape store images
    :param image_path: the place to store images
    """
    files = sorted(list(get_file_paths(path=cityscapes_image_folder)))
    os.makedirs(image_path, exist_ok=True)
    for f in files:
        filename = re.split('/', f)[-1]
        copyfile(f, os.path.join(image_path, filename))


def main():
    # count_files(path='./leftImg8bit/test')

    # annotation = Annotation()
    # annotation.fromJsonFile('./gtBboxCityPersons/train/aachen/aachen_000000_000019_gtBboxCityPersons.json')
    # for obj in annotation.objects:
    #     print(obj)

    # show_image_annotated('./gtBboxCityPersons/train/aachen/aachen_000000_000019_gtBboxCityPersons.json',
    #                     './leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png')
    # show_image_annotated('./gtBboxCityPersons/train/aachen/aachen_000001_000019_gtBboxCityPersons.json',
    #                      './leftImg8bit/train/aachen/aachen_000001_000019_leftImg8bit.png')
    # label_count('./gtBboxCityPersons/train/')
    # label_count('./gtBboxCityPersons/val/')

    os.chdir('../cityscapes')
    print("Current Working Directory ", os.getcwd())

    generate_yolo_label_files(cityscapes_annotation_folder='./gtBboxCityPersons/train/', yolo_annotation_folder='.', name='train')
    generate_yolo_label_files(cityscapes_annotation_folder='./gtBboxCityPersons/val/', yolo_annotation_folder='.', name='val')

    generate_yolo_image_path_file(cityscapes_image_folder='./leftImg8bit/train/', yolo_image_file='./train.txt', image_path='./images/train')
    generate_yolo_image_path_file(cityscapes_image_folder='./leftImg8bit/val/', yolo_image_file='./val.txt', image_path='./images/val')
    generate_yolo_image_path_file(cityscapes_image_folder='./leftImg8bit/test/', yolo_image_file='./test.txt', image_path='./images/test')

    create_image_folder(cityscapes_image_folder='./leftImg8bit/train/', image_path='./images/train')
    create_image_folder(cityscapes_image_folder='./leftImg8bit/val/', image_path='./images/val')
    create_image_folder(cityscapes_image_folder='./leftImg8bit/test/', image_path='./images/test')

    pass
if __name__ == '__main__':
    main()
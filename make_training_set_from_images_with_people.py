import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))

import darknet as dn
import pdb
import shutil
import cv2

human_dir = './predicted_images'

font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 0.6
fontColor              = (238,20,20)
lineType               = 2

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def tup2int(tup):
    return tuple(int(_) for _ in tup)

class BBox:
    def __init__(self, orgX, orgY, width, height, category, confidence):
        self.origin = (int(orgX), int(orgY))
        self.w = width
        self.h = height
        self.x1 = int(orgX - width/2)
        self.y1 = int(orgY - height/2)
        self.x2 = int(orgX + width/2)
        self.y2 = int(orgY + height/2)
        self.category = category
        self.confidence = confidence

    def __getitem__(self, item):
        return getattr(self, item)
    
    @classmethod
    def from_darknet(cls, dn_output_row):
        category, confidence, (centerX, centerY, width, height) = dn_output_row
        return cls(orgX=centerX, orgY=centerY, width=width, height=height, category=category, confidence=confidence)

    def draw(self, img):
        cv2.rectangle(img, (self.x1, self.y1), (self.x2, self.y2), 19)
        cv2.putText(img, self.category, 
            (self.x1, self.y1), 
            font, 
            fontScale,
            fontColor,
            lineType)

dn.set_gpu(0)
net = dn.load_net("cfg/yolov3.cfg", "yolov3.weights", 0)
meta = dn.load_meta("cfg/coco.data")

r = dn.detect(net, meta, "data/dog.jpg")
print(r)
bboxes = [BBox.from_darknet(row) for row in r]
img = cv2.imread("data/dog.jpg")

for bbox in bboxes:
    bbox.draw(img)
cv2.imshow('', img)
cv2.waitKey(0)
# x, y, width, height
#coords = (375.4502868652344, 268.9792785644531, 515.8342895507812, 353.4102783203125)
#x, y, width, height = coords
#pt1 = (x - width / 2, y - height / 2)
#pt2 = (x + width / 2, y + height / 2)
#cv2.rectangle(img,tup2int(pt1) ,tup2int(pt2), 8)
#cv2.imshow('', img)
#cv2.waitKey(0)


#def num_people(prediction):
#    return [obj for obj in prediction if 'person' in obj]
#
#for image_name in os.listdir(human_dir):
#    print(image_name)

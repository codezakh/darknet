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


class Config:
    bad_images_dir = './bad'
    good_images_dir = './good'
    category_no_human = './good/no-person'
    category_human = './good/person'
    name = "{orig_img_name}-partition_{partition_idx}.{extension}"
    original_image_dir = './predicted_images'
    extension = 'jpg'

    @classmethod
    def name_fmt(self, orig_img_name, partition_idx):
        return self.name.format(orig_img_name=orig_img_name,
                partition_idx=partition_idx, extension=self.extension)


def partition_human_image(image):
    """
    Partition images into 6 sectors, 2 rows and 3 columns.
    The first dimension is the y, the second is the x. The
    partitions in this have the same order as the bounding 
    boxes that come out of `get_bboxes_for_partition`.
    """
    # images are 480 tall and 640 wide
    partition_one = image[0:224, 0:224]
    partition_two = image[0:224, 224:448]
    partition_three = image[0:224, 416:640]

    partition_four = image[224:448, 0:224]
    partition_five = image[224:448, 224:448]
    partition_six = image[224:448, 416:640]

    return (partition_one, partition_two, partition_three,
            partition_four, partition_five, partition_six)

def get_bboxes_for_partition():
    """
    Assume the image is 480 x 640. The bottom part of the image is cut
    off, it isn't important since it seems to be blocked by the casing. 
    There is overlap on the horizontal parts because 224 x 3> 640.

    Return
    ----------
    A list of tuples of the form (p1, p2), where p1 is the top left corner,
    and p2 is the bottom right corner of a rectangle. Both are represented in standard
    cartesian notation.
    """
    partition_one = ( (0,0), (224, 224) )
    partition_two = ( (224, 0), (448, 224))
    partition_three = ( (416, 0), (640, 224) )

    partition_four = ( (0,224), (224, 448) )
    partition_five = ( (224, 224), (448, 448))
    partition_six = ( (416, 224), (640, 448) )

    return (partition_one, partition_two, partition_three,
            partition_four, partition_five, partition_six)


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
    #iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    #assert iou >= 0.0
    #assert iou <= 1.0
    #return iou

    # return the intersection divided by bb2's area, because this function will
    # really only be used to check how much a bounding box intersects with a ROI
    return intersection_area / float(bb2_area)

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

    def get_pct_of_self_in_other(self, otherbbox):
        return get_iou(otherbbox, self)
    
    @classmethod
    def from_darknet(cls, dn_output_row):
        category, confidence, (centerX, centerY, width, height) = dn_output_row
        return cls(orgX=centerX, orgY=centerY, width=width, height=height, category=category, confidence=confidence)

    def draw(self, img):
        cv2.rectangle(img, (self.x1, self.y1), (self.x2, self.y2), (0, 255, 0))
        cv2.putText(img, self.category, 
            (self.x1, self.y1), 
            font, 
            fontScale,
            fontColor,
            lineType)

dn.set_gpu(0)
net = dn.load_net("cfg/yolov3.cfg", "yolov3.weights", 0)
meta = dn.load_meta("cfg/coco.data")

for image_name in os.listdir(Config.original_image_dir):
    image_full_path = os.path.join(Config.original_image_dir, image_name)
    img = cv2.imread(image_full_path)
    r = dn.detect(net, meta, image_full_path)
    bboxes = [BBox.from_darknet(row) for row in r]
    human_bboxes = [bbox for bbox in bboxes if bbox.category == 'person']
    partitions = partition_human_image(img)
    print(image_full_path)
    for idx, (p1, p2) in enumerate(get_bboxes_for_partition()):
        tmp_virtual_bbox = dict(x1=p1[0], y1=p1[1], x2=p2[0], y2=p2[1])
        intersects_with_person = any([person.get_pct_of_self_in_other(tmp_virtual_bbox) >= 0.3 for person in
                human_bboxes])
        if intersects_with_person:
            path_prefix = Config.category_human
        else:
            path_prefix = Config.category_no_human

        partition_write_path = os.path.join(path_prefix, Config.name_fmt(image_name, idx))
        cv2.imwrite(partition_write_path, partitions[idx])
sys.exit(0)




r = dn.detect(net, meta, "predicted_images/ifm-ips-01_1540306640.jpg")
print(r)
bboxes = [BBox.from_darknet(row) for row in r]
img = cv2.imread("predicted_images/ifm-ips-01_1540306640.jpg")

for bbox in bboxes:
    bbox.draw(img)

human_bboxes = [bbox for bbox in bboxes if bbox.category == 'person']
partitions = partition_human_image(img)
for idx, (p1, p2) in enumerate(get_bboxes_for_partition()):
    tmp_virtual_bbox = dict(x1=p1[0], y1=p1[1], x2=p2[0], y2=p2[1])
    intersects_with_person = any([person.get_pct_of_self_in_other(tmp_virtual_bbox) >= 0.3 for person in
            human_bboxes])
    cv2.rectangle(img, p1, p2, (255, 0, 0) if intersects_with_person else (0, 0, 255))
    if intersects_with_person:
        cv2.imshow(str(idx), partitions[idx])
cv2.imshow('', img)
cv2.waitKey(0)

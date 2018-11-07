import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))

import darknet as dn
import pdb
import shutil

raw_data_dir_abspath = "/home/ifm/human_detector/data/raw_data_102218"

human_dir = './predicted_images'

dn.set_gpu(0)
net = dn.load_net("cfg/yolov3.cfg", "yolov3.weights", 0)
meta = dn.load_meta("cfg/coco.data")

r = dn.detect(net, meta, "data/eagle.jpg")
print(r)

def num_people(prediction):
    return [obj for obj in prediction if 'person' in obj]

count = 0 
for root, dirs, files in os.walk(raw_data_dir_abspath):
    for name in files:
        if os.path.isdir(os.path.join(root, name)):
            continue
        else:
            r = dn.detect(net, meta, os.path.join(root, name))
            n_ppl_in_image = num_people(r)
            if len(n_ppl_in_image) >= 2:
                print(r)
                print('Greater than 2 people detected, copying image')
                print("Found {} training images.".format(count))
                shutil.copyfile(os.path.join(root, name), os.path.join(human_dir, name))
                count += 1

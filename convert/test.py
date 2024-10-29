import cv2

img = cv2.imread("/home/hzf/data/Refer-Youtube_VIS/train/JPEGImages/0043f083b5/00040.jpg")
rec = [1153, 372, 21, 54]
draw_0  = cv2.rectangle(img, rec, (255, 0, 0), 2)
cv2.imwrite('test.jpg', draw_0)

# import json

# anno = json.load(open("/home/hzf/data/Refer-Youtube_VIS/train/train.json", 'r'))
# print()

# import os
# path = "/home/hzf/data/Refer-Youtube_VIS/train/JPEGImages/"
# dicts = os.listdir(path)
# count = 0
# for one_dict in dicts:
#     count += 1
# print(count)
import cv2
import os.path
import glob
import numpy as np
from PIL import Image

def convertPNG(pngfile, outdir):
	# 读取灰度图
	im_depth = cv2.imread(pngfile)
	# 转换成伪彩色（之前必须是8位图片）
	# 这里有个alpha值，深度图转换伪彩色图的scale可以通过alpha的数值调整，我设置为1，感觉对比度大一些
	im_color = cv2.applyColorMap(cv2.convertScaleAbs(im_depth, alpha=1), cv2.COLORMAP_JET)
	# 转成png
	im = Image.fromarray(im_color)
	# 保存图片
	im.save(os.path.join(outdir, os.path.basename(pngfile)))

for pngfile in glob.glob("/home/hzf/project/46/MOTRv2/attmap/*.jpg"):
	convertPNG(pngfile, "/home/hzf/project/46/MOTRv2/colormap/")

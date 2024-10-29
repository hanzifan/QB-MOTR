import cv2
import os

# 设置输入图像的目录和输出视频的名称
input_folder = '/home/hzf/project/46/MOTRv2/exps/default/b0psypDj4_c006/'
output_file = 'b0psypDj4_c006.avi'

# 获取输入文件夹中的所有文件
image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]
image_files.sort(key=lambda x:int(x.split('.')[0][6:]))

# 获取第一张图像的大小
img = cv2.imread(os.path.join(input_folder, image_files[0]))
height, width, layers = img.shape

# 创建视频编码器
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(output_file, fourcc, 20, (width, height))

# 循环遍历所有图像，并将它们添加到视频中
for image in image_files:
    img = cv2.imread(os.path.join(input_folder, image))
    video.write(img)

# 释放视频资源
# cv2.destroyAllWindows()
# video.release()
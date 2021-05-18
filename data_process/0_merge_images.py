"""
将widerface的分散在各个文件夹里的图片汇总到一起
"""

import os
import sys

src_folder_path=sys.argv[1]
dst_folder_path=src_folder_path.replace('images','combined_images')
os.makedirs(dst_folder_path,exist_ok=True)

folder_names=os.listdir(src_folder_path)
folder_names.sort()

for i,folder_name in enumerate(folder_names):
    print("--------------process [{}/{}]:{}----------------".format(i+1,len(folder_names),folder_name))
    folder_path=os.path.join(src_folder_path,folder_name)
    img_names=os.listdir(folder_path)

    for j,img_name in enumerate(img_names):
        src_img_path=os.path.join(folder_path,img_name)
        dst_img_path=os.path.join(dst_folder_path,img_name)
        os.system("cp {} {}".format(src_img_path,dst_img_path))
        if j%100==0:
            print("finished {}/{}".format(j,len(img_names)))




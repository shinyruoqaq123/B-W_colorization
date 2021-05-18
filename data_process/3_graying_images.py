import os
import sys
import cv2

src_folder_path=sys.argv[1]
folder_name=src_folder_path.split(os.path.sep)[-1]
dst_folder_path=src_folder_path.replace(folder_name,'{}_gray'.format(folder_name))
os.makedirs(dst_folder_path,exist_ok=True)
img_names=os.listdir(src_folder_path)
for i,img_name in enumerate(img_names):
    src_img_path=os.path.join(src_folder_path,img_name)
    dst_img_path=os.path.join(dst_folder_path,img_name)

    src_img=cv2.imread(src_img_path)
    dst_img=cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(dst_img_path,dst_img)

    if i%1000==0:
        print("finished {}/{}".format(i,len(img_names)))
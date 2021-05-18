import os
import sys

data_folder_path=sys.argv[1]
folder_name=data_folder_path.split(os.path.sep)[-1]
dst_file_path='../dataset/animals_{}}.txt'.format(folder_name)
dst_file=open(dst_file_path,'w')
total_count=0

img_names=os.listdir(data_folder_path)
for img_name in img_names:
    img_path = os.path.join(data_folder_path, img_name)
    dst_file.writelines("{}\n".format(img_path))
    total_count += 1

print("Done, total_count={}".format(total_count))
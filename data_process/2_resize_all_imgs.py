from multiprocessing import Pool
from PIL import Image
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Resize all colorful imgs to 256*256 for training")
    parser.add_argument("-d",
                        "--dir",
                        required=True,
                        type=str,
                        help="The directory includes all jpg images")
    parser.add_argument("-n",
                        "--nprocesses",
                        default=10,
                        type=int,
                        help="Using how many processes")
    args = parser.parse_args()
    return args

def doit(x):
    a=Image.open(x)
    if a.getbands()!=('R','G','B'):
        os.remove(x)
        return
    folder_name=x.split(os.path.sep)[-2]
    result_path=x.replace(folder_name,'{}_256'.format(folder_name))
    a.resize((256,256),Image.BICUBIC).save(result_path)
    return

args=parse_args()

folder_name=args.dir.split(os.path.sep)[-1]
dst_folder_path=args.dir.replace(folder_name,'{}_256'.format(folder_name))
os.makedirs(dst_folder_path,exist_ok=True)

flist = os.listdir(args.dir)

for i,f_name in enumerate(flist):
    doit(os.path.join(args.dir,f_name))
    if i%100==0:
        print("finished {}/{}".format(i,len(flist)))
print('done')
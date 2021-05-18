from skimage.measure import compare_ssim
from skimage import io
import sys
import os

generated_folder_path=sys.argv[1]
gt_folder_path=sys.argv[2]

img_names=os.path.join(generated_folder_path)
total_ssim=0
for img_name in img_names:
    generated_img_path=os.path.join(generated_folder_path,img_name)
    gt_img_path=os.path.join(gt_folder_path,img_name)

    generated_img=io.imread(generated_img_path)
    gt_img=io.imread(gt_img_path)

    ssim=compare_ssim(generated_img,gt_img,multichannel=True)
    total_ssim+=ssim

print("average SSIM={:.3f}".format(total_ssim/len(img_names)))

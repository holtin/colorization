import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2
import colour



imgpath ="/content/drive/MyDrive/Learning/color/movie/movie_dataset"
paths = glob.glob(imgpath + "/*.jpg")


output_num=round(len(paths)*0.1)
np.random.seed(123)
paths_subset = np.random.choice(paths, output_num, replace=False) 
count=0

# load colorizers
#colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()


# default size to process images is 256x256
# grab L channel in both original ("orig") and resized ("rs") resolutions
print("testing size")
print(len(paths_subset))
average_deltae=0
for link in paths_subset:
    count=count+1
    img = load_img(link)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
    # colorizer outputs 256x256 ab map
    # resize and concatenate to original L channel
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
    #out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

    normalized_array = (out_img_siggraph17 - np.min(out_img_siggraph17))/(np.max(out_img_siggraph17) - np.min(out_img_siggraph17)) # this set the range from 0 till 1
    img_n = (normalized_array * 255).astype(np.uint8) # set is to a range from 0 till 255

    img2 = Image.fromarray(img_n, 'RGB')
    img2.save('tmp.jpg')
    colorized_img = cv2.imread('tmp.jpg')
    ori_img = cv2.imread(link)
    image1_lab = cv2.cvtColor(ori_img, cv2.COLOR_RGB2Lab)
    image2_lab = cv2.cvtColor(colorized_img, cv2.COLOR_RGB2Lab)
    delta_E = colour.delta_E(image1_lab, image2_lab)
    average_deltae=average_deltae+np.mean(delta_E)
    print(average_deltae/count)

#plt.imsave('%s_eccv16.png'%opt.save_prefix, out_img_eccv16)
#plt.imsave('%s_siggraph17.png'%opt.save_prefix, out_img_siggraph17)
import glob
import colour
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from PIL import Image, ImageOps
from skimage.color import lab2rgb, rgb2lab
from torchvision import transforms
from model import MainModel,build_res_unet,lab_to_rgb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
compare=True
display=False
export=True

imgpath ="./predict_img"
modelpath="./model/gen200_m20.pth"
pathOut="./export"

paths = glob.glob(imgpath + "/*.jpg")
print("length")
print(len(paths))
output_num=round(len(paths)*0.1)
np.random.seed()
paths_subset = np.random.choice(paths, output_num, replace=False) 
count=0

if __name__ == '__main__':
    net_G = build_res_unet(n_input=1, n_output=2, size=256)    
    model = MainModel(net_G=net_G)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(
        torch.load(
            modelpath,
            map_location=device
        )
    )
    
    for link in paths_subset:
        count=count+1
        img = PIL.Image.open(link)
        img2= ImageOps.grayscale(img)
        im = transforms.ToTensor()(img2)[:1] * 2. - 1.
        model.eval()
        with torch.no_grad():
            preds = model.net_G(im.unsqueeze(0).to(device))
            colorized = lab_to_rgb(im.unsqueeze(0), preds.cpu())[0]
            normalized_array = (colorized - np.min(colorized))/(np.max(colorized) - np.min(colorized)) # this set the range from 0 till 1
            img_n = (normalized_array * 255).astype(np.uint8) # set is to a range from 0 till 255
        
        if compare:

            img2 = Image.fromarray(img_n, 'RGB')
            img2.save('tmp.jpg')
            colorized_img = cv2.imread('tmp.jpg')
            ori_img = cv2.imread(link)
            image1_lab = cv2.cvtColor(ori_img, cv2.COLOR_RGB2Lab)
            image2_lab = cv2.cvtColor(colorized_img, cv2.COLOR_RGB2Lab)
            delta_E = colour.delta_E(image1_lab, image2_lab)
            print(np.mean(delta_E))

        try:
            
            if display:
                plt.imshow(img)
                plt.show()
                plt.imshow(colorized)
                plt.show()
            
            if export:
                img2 = Image.fromarray(img_n, 'RGB')
                img2.save(pathOut + "/%d_predict.jpg" % count)
                

        except Exception:
            pass

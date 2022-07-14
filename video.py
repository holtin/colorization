import cv2
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from model import MainModel,build_res_unet,lab_to_rgb

fps = 24
sec = 290
skipsec=260

model_path="./model/gen200_m20.pth"
video_path="./movie/video5.mp4"


fourcc = cv2.VideoWriter_fourcc(*'mp4v')

demo= False
frame=1

if __name__ == '__main__':
    net_G = build_res_unet(n_input=1, n_output=2, size=256)    
    model = MainModel(net_G=net_G)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    if demo:
      frame=60

    #target video
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    height, width, channels = image.shape
    if not demo:
      video = cv2.VideoWriter('color_video5_out.mp4', fourcc, float(fps), (width, height))

    for i in range (fps*sec):
      success,image = vidcap.read()

      if i < fps*skipsec:
        pass
      
      elif i%frame==0:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        im_pil = Image.fromarray(img)
        im_pil =transforms.ToTensor()(im_pil)[:1] * 2. - 1.
        model.eval()
        
        with torch.no_grad():
          preds = model.net_G(im_pil.unsqueeze(0).to(device))
          colorized = lab_to_rgb(im_pil.unsqueeze(0), preds.cpu())[0]
          #print(colorized.dtype)
          normalized_array = (colorized - np.min(colorized))/(np.max(colorized) - np.min(colorized)) # this set the range from 0 till 1
          img_n = (normalized_array * 255).astype(np.uint8) # set is to a range from 0 till 255

          if demo:
            plt.imshow(img_n)
            plt.show()
          else:
            img2 = Image.fromarray(img_n, 'RGB')
            img2.save('holtinmy.png')
            img3 = cv2.imread('holtinmy.png')
            print(i)
            video.write(img3)
    if not demo:
      video.release()
      print('video released')
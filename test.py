import PIL
import colour
import cv2
from PIL import Image, ImageChops,ImageStat

#path ="/content/drive/MyDrive/Learning/color/coco_sample"
model_path="/content/drive/MyDrive/Learning/color/new_test8020_100_m20.pth"

path="/content/drive/MyDrive/Learning/color/movie/movie_dataset"
#path="/content/drive/MyDrive/Learning/color/coco_sample"
paths = glob.glob(path + "/*.jpg")
ds_size=round(len(paths)*0.05)
np.random.seed()
paths_subset = np.random.choice(paths, 5, replace=False) 
count=0
export=False
pathOut="/content/drive/MyDrive/Learning/color/predict_result/tmp/"
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
    
    print("length of dataset is")
    print(len(paths_subset))
    avg_de=0
    fig = plt.figure(figsize=(15, 8))
    for link in paths_subset:
      
      #path = "/content/drive/MyDrive/Learning/color/test12.jpg"
      img = PIL.Image.open(link)
      img2= ImageOps.grayscale(img)
      im = transforms.ToTensor()(img2)[:1] * 2. - 1.
      model.eval()
      with torch.no_grad():
          preds = model.net_G(im.unsqueeze(0).to(device))
      colorized = lab_to_rgb(im.unsqueeze(0), preds.cpu())[0]

      img3 = load_img(link)
      (tens_l_orig, tens_l_rs) = preprocess_img(img3, HW=(256,256))
      # colorizer outputs 256x256 ab map
      # resize and concatenate to original L channel
      img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
      #out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
      out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())


      
      ax = plt.subplot(3, 5, count + 1)
      
      ax.imshow(img_bw)
      ax.axis("off")
      ax = plt.subplot(3, 5, count + 1 + 5)
      
      ax.imshow(colorized)
      ax.axis("off")
      ax = plt.subplot(3, 5, count + 1 + 10)
      ax.imshow(out_img_siggraph17)
      ax.axis("off")
      count=count+1

    plt.show()
    fig.savefig("result_m3.png")
import glob
import numpy as np
import torch
from PIL import Image
from skimage.color import rgb2lab
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm.notebook import tqdm
from model import MainModel,build_res_unet,create_loss_meters,update_losses,log_results,visualize


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
ds_path ='./movie_dataset'
old_mod_path="./model/gen200_m20.pth"
new_mod_path="./model/gen200_m25.pth"
ds_size=8800
SIZE = 256

class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train'):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE),  Image.BICUBIC),
                transforms.RandomHorizontalFlip(), # A little data augmentation!
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((SIZE, SIZE),  Image.BICUBIC)
        
        self.split = split
        self.size = SIZE
        self.paths = paths
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32") # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1
        
        return {'L': L, 'ab': ab}
    
    def __len__(self):
        return len(self.paths)

def make_dataloaders(batch_size=16, n_workers=0, pin_memory=True, **kwargs): # A handy function to make our dataloaders
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader
 
def train_model(model, train_dl, epochs, display_every=200):
    #data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
    for e in range(epochs):
        print(e)
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to 
        
        i = 0                                  # log the losses of the complete network
        for data in tqdm(train_dl):
            
            model.setup_input(data) 
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                log_results(loss_meter_dict) # function to print out the losses
                visualize(model, data, save=False) # function displaying the model's outputs


paths = glob.glob(ds_path + "/*.jpg") # Grabbing all the image file names
print(len(paths))
np.random.seed()
paths_subset = np.random.choice(paths, ds_size, replace=False) # choosing 20000 images randomly
rand_idxs = np.random.permutation(ds_size)
train_idxs = rand_idxs[:ds_size] # choosing the first 8000 as training set
val_idxs = rand_idxs[ds_size:] # choosing last 2000 as validation set
train_paths = paths_subset[train_idxs]
val_paths = paths_subset[val_idxs]
print(len(train_paths), len(val_paths))

train_dl = make_dataloaders(paths=train_paths, split='train')
val_dl = make_dataloaders(paths=val_paths, split='val')

data = next(iter(train_dl))
Ls, abs_ = data['L'], data['ab']
print(Ls.shape, abs_.shape)
print(len(train_dl), len(val_dl))

net_G = build_res_unet(n_input=1, n_output=2, size=256)
model = MainModel(net_G=net_G)
model.load_state_dict(
        torch.load(
            old_mod_path,
            map_location=device
        )
    )
train_model(model, train_dl, 5,200)
torch.save(model.state_dict(), new_mod_path)
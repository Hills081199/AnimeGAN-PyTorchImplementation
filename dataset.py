import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import normalize_input, compute_data_mean

class AnimeDataset(Dataset):
    def __init__(self, data_dir, anime_dir, transform=None):
        self.train_photos_dir = os.path.join(data_dir, "train_photo")
        self.anime_style_dir = os.path.join(data_dir, anime_dir + "/style")
        self.anime_smooth_dir = os.path.join(data_dir, anime_dir + "/smooth")
        
        self.files = {}
        self.files['train_imgs'] = [os.path.join(self.train_photos_dir, file) for file in os.listdir(self.train_photos_dir)]
        self.files['anime_style_imgs'] = [os.path.join(self.anime_style_dir, file) for file in os.listdir(self.anime_style_dir)]
        self.files['anime_smooth_imgs'] = [os.path.join(self.anime_smooth_dir, file) for file in os.listdir(self.anime_smooth_dir)]
        self.num_train_imgs = len(self.files['train_imgs'])
        self.num_anime_imgs = len(self.files['anime_style_imgs'])
        print(self.num_train_imgs, self.num_anime_imgs)
        
        self.mean = compute_data_mean(os.path.join(data_dir, anime_dir + "/style"))
        self.transform = transform
    
    def __len__(self):
        return len(self.files['train_imgs'])
    
    def __getitem__(self, idx):
        img = self.load_train_img(idx)
        anime_idx = idx
        if anime_idx > self.num_anime_imgs:
            anime_idx -= self.num_anime_imgs * (idx//self.num_anime_imgs)
        
        anime_style, anime_style_gray = self.load_anime(anime_idx)
        anime_smooth = self.load_anime_smooth(anime_idx)

        return img, anime_style, anime_style_gray, anime_smooth

    def load_train_img(self, idx):
        img_path = self.files['train_imgs'][idx]   
        img = cv2.imread(img_path)[:,:,::-1] #BGR -> RGB
        img = self.take_transform(img, add_mean=False)
        img = img.transpose(2, 0, 1)
        return torch.tensor(img)
    
    def load_anime(self, idx):
        img_path = self.files['anime_style_imgs'][idx]
        img = cv2.imread(img_path)[:,:,::-1]
        img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        img_gray = np.stack([img_gray, img_gray, img_gray], axis=-1)
        img_gray = self.take_transform(img_gray, add_mean=False)
        img_gray = img_gray.transpose(2, 0, 1)

        img = self.take_transform(img, add_mean=True)
        img = img.transpose(2, 0, 1)
        
        return torch.Tensor(img), torch.Tensor(img_gray)

    def load_anime_smooth(self, idx):
        img_path = self.files['anime_smooth_imgs'][idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = np.stack([img, img, img], axis=-1)
        img = self.take_transform(img, add_mean=False)
        img = img.transpose(2, 0, 1)
        return torch.tensor(img)

    def take_transform(self, img, add_mean=True):
        if self.transform is not None:
            img = self.transform(img)
        
        img = img.astype(np.float32)
        if add_mean:
            img += self.mean
        
        return normalize_input(img)


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     from torch.utils.data import DataLoader

#     data_dir = "D:\Datasets\\animegan_dataset"
#     anime_dir = "D:\Datasets\\animegan_dataset\Hayao"
#     ds = AnimeDataset(data_dir, anime_dir)
#     #anime_loader = DataLoader(AnimeDataset(data_dir, anime_dir), batch_size=2, shuffle=True)

#     img, anime_style, anime_gray, anime_smooth = ds[0]
#     plt.imshow(img.numpy().transpose(1, 2, 0))
#     plt.show()
#     plt.imshow(anime_style.numpy().transpose(1,2,0))
#     plt.show()
#     plt.imshow(anime_gray.numpy().transpose(1,2,0))
#     plt.show()
#     plt.imshow(anime_smooth.numpy().transpose(1,2,0))
#     plt.show()
    

# if __name__ == '__main__':
#     data_dir = "D:\Datasets\\animegan_dataset"
#     anime_dir = "D:\Datasets\\animegan_dataset\Hayao"
#     ds = AnimeDataset(data_dir, anime_dir)
#     ds[0]
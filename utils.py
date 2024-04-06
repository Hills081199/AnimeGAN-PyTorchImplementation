import os
import torch
import numpy as np
import cv2
from tqdm import tqdm

rgb_to_yuv_kernel = torch.tensor([
    [0.299, -0.14714119, 0.61497538],
    [0.587, -0.28886916, -0.51496512],
    [0.114, 0.43601035, -0.10001026]
]).float

def normalize_input(images):
    '''
    [0, 255] -> [-1, 1]
    '''
    return images / 127.5 - 1.0

def compute_data_mean(data_folder):
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f'Folder {data_folder} does not exits')

    image_files = os.listdir(data_folder)
    total = np.zeros(3)

    print(f"Compute mean (R, G, B) from {len(image_files)} images")

    for img_file in tqdm(image_files):
        path = os.path.join(data_folder, img_file)
        image = cv2.imread(path)
        total += image.mean(axis=(0, 1))

    channel_mean = total / len(image_files)
    mean = np.mean(channel_mean)

    return mean - channel_mean[...,::-1]  # Convert to BGR for training

def rgb_to_yuv(img):
    img = (img+1.0)/2.0
    if torch.cuda.is_available():
        rgb_to_yuv_kernel = rgb_to_yuv_kernel.cuda()
    yuv = torch.tensordot(img, rgb_to_yuv_kernel, dims=([img.ndim-3],[0]))
    return yuv

def gram_matrix(input):
    """
    https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#style-loss
    """

    batch_size, channels, width, height = input.size()
    features = input.view(batch_size * channels, width * height)

    #Gram product
    G = torch.mm(features, features.t())

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(batch_size*channels*width*height)

def gaussian_noise():
    return torch.normal(torch.tensor(0.0), torch.tensor(0.1))

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_model(model_name, model_dir, model, optimizer, epoch):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    } 
    save_path = os.path.join(model_dir, f'{model_name}.pth')
    torch.save(checkpoint, save_path)

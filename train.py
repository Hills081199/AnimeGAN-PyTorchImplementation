import train_config
import argparse
import torch

from torch.utils.data import DataLoader
from model.generator import Generator
from model.discriminator import Discriminator
from dataset import AnimeDataset
from utils import set_lr, save_model
from loss import AnimeLoss

if __name__ == '__main__':
    #Get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("DEVICE : ", device)

    #dataset and dataloader
    ds = AnimeDataset(data_dir = train_config.data_dir, anime_dir=train_config.anime_dir)
    data_loader = DataLoader(ds, batch_size=train_config.batch_size, shuffle=True)

    #Load model
    G = Generator().to(device)
    D = Discriminator().to(device)

    if train_config.starting_training_epoch != 0:
        assert train_config.starting_training_epoch < train_config.total_epochs and train_config.starting_training_epoch >= 0, "starting training epoch should be less than total epochs and greater than 0"
        
        #load existed model state dict to continue training

    #optimizer
    optimizer_g = torch.optim.Adam(G.parameters(), train_config.learning_rate, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(D.parameters(), train_config.learning_rate, betas=(0.5, 0.999))

    #loss
    anime_loss = AnimeLoss(device, train_config.wadvg, train_config.wadvd, train_config.wcon, train_config.wgra, train_config.wcol)

    print("Start training from epoch : ", train_config.starting_training_epoch)
    for epoch in range(train_config.starting_training_epoch, train_config.total_epochs):
        content_loss_epoch = []
        gram_loss_epoch = []
        color_loss_epoch = []
        adversarial_g_loss_epoch = []
        g_loss_epoch = []
        d_loss_epoch = []

        """
        suggests that the pre-training of the generator helps to accelerate
        GAN convergence. Hence, the generator network G is pre-trained with only the
        content loss Lcon(G, D). The initialization training is performed for one epoch
        and the learning rate is set to 0.0001.
        """
        if epoch < train_config.starting_training_epoch:
            set_lr(optimizer_g, train_config.init_learning_rate_g)
            for img, *_ in data_loader:
                img = img.to(device)
                optimizer_g.zero_grad()
                fake_img = G(img)
                content_loss = anime_loss.l_con_init(img, fake_img)
                content_loss.backward()
                optimizer_g.step()
                
                content_loss_epoch.append(content_loss.cpu().detach().numpy())
            
            init_content_loss = sum(content_loss_epoch) / len(content_loss_epoch)
            print("Epoch : {}/{}, content_loss : {}".format(epoch+1, train_config.total_epochs, init_content_loss))

            #save init model
            save_model(model_name="G_init", model_dir=train_config.model_dir, model=G, optimizer=optimizer_g, epoch=epoch+1)
            continue

        # for img, anime_style, anime_style_gray, anime_smooth in data_loader:
        #     img = img.to(device)
        #     anime_style = anime_style.to(device)
        #     anime_style_gray = anime_style_gray.to(device)
        #     anime_smooth = anime_smooth.to(device)

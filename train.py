import train_config
import os
import torch

from torch.utils.data import DataLoader
from model.generator import Generator
from model.discriminator import Discriminator
from dataset import AnimeDataset
from utils import set_lr, save_model, load_model, gaussian_noise, save_samples
from loss import AnimeLoss
from tqdm import tqdm
import time

torch.autograd.set_detect_anomaly(True)

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

    if train_config.starting_training_epoch <= train_config.init_g_starting_epoch:
        train_config.starting_training_epoch = 0
    
    else:
        g_model_name = f'G_{train_config.anime_dataset}_at_epoch_{train_config.starting_training_epoch+1}'
        d_model_name = f''f'D_{train_config.anime_dataset}_at_epoch_{train_config.starting_training_epoch+1}'
        g_checkpoint_existed_path = os.path.join(train_config.model_dir, f'{g_model_name}.pth')
        d_checkpoint_existed_path = os.path.join(train_config.model_dir, f'{d_model_name}.pth')
        if not os.path.exists(g_checkpoint_existed_path) or not os.path.exists(d_checkpoint_existed_path):
            raise FileNotFoundError(f'Not found existed model G / D at epoch : {train_config.starting_training_epoch+1}')
        
        print("LOAD existed G ...")
        load_model(g_model_name, train_config.model_dir, G)

        print("LOAD existed D ...")
        load_model(d_model_name, train_config.model_dir, D)
        
        #load existed model state dict to continue training

    #optimizer
    optimizer_g = torch.optim.Adam(G.parameters(), train_config.learning_rate_g, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(D.parameters(), train_config.learning_rate_d, betas=(0.5, 0.999))

    #loss
    anime_loss = AnimeLoss(device, train_config.wadvg, train_config.wadvd, train_config.wcon, train_config.wgra, train_config.wcol)

    if not os.path.isdir(train_config.model_dir):
        os.makedirs(train_config.model_dir)

    if not os.path.isdir(train_config.save_img_dir):
        os.makedirs(train_config.save_img_dir)
    
    print("Start training from epoch : ", train_config.starting_training_epoch)
    for epoch in range(train_config.starting_training_epoch, train_config.total_epochs):
        init_content_loss_epoch = []
        # content_loss_epoch = []
        # gram_loss_epoch = []
        # color_loss_epoch = []
        # adversarial_g_loss_epoch = []
        g_loss_epoch = []
        d_loss_epoch = []

        """
        suggests that the pre-training of the generator helps to accelerate
        GAN convergence. Hence, the generator network G is pre-trained with only the
        content loss Lcon(G, D). The initialization training is performed for one epoch
        and the learning rate is set to 0.0001.
        """
        if epoch < train_config.init_g_starting_epoch:
            print("INIT SESSION....")
            set_lr(optimizer_g, train_config.init_learning_rate_g)
            for img, *_ in tqdm(data_loader):
                img = img.to(device)
                optimizer_g.zero_grad()
                fake_img = G(img)
                content_loss = anime_loss.l_con_init(img, fake_img)
                content_loss.backward()
                optimizer_g.step()
                
                init_content_loss_epoch.append(content_loss.cpu().detach().numpy())
                
            init_content_loss = sum(init_content_loss_epoch) / len(init_content_loss_epoch)
            print("Epoch : {}/{}, content_loss_init : {}".format(epoch+1, train_config.total_epochs, init_content_loss))

            #save init model
            save_model(model_name="G_init_at_epoch_{}".format(epoch+1), model_dir=train_config.model_dir, model=G, optimizer=optimizer_g, epoch=epoch+1)
            continue
            
        for img, anime_style, anime_style_gray, anime_smooth in tqdm(data_loader):
            img = img.to(device)
            anime_style = anime_style.to(device)
            anime_style_gray = anime_style_gray.to(device)
            anime_smooth = anime_smooth.to(device)

            #train D
            optimizer_d.zero_grad()
            fake_img = G(img).detach()
            fake_img += gaussian_noise()
            anime_style += gaussian_noise()
            anime_style_gray += gaussian_noise()
            anime_smooth += gaussian_noise()

            fake_logit = D(fake_img)
            real_anime_logit = D(anime_style)
            real_anime_gray_logit = D(anime_style_gray)
            real_anime_smooth_gray_logit = D(anime_smooth)
            d_loss = anime_loss.d_loss(real_anime_logit, fake_logit, real_anime_gray_logit, real_anime_smooth_gray_logit)
            d_loss.backward()
            optimizer_d.step()

            d_loss_epoch.append(d_loss)

            #train G
            optimizer_g.zero_grad()
            fake_img = G(img)
            fake_logit = D(fake_img)
            adversarial_g_loss, content_loss, gram_loss, color_loss = anime_loss.g_loss(img, fake_img, fake_logit, anime_style_gray)
            g_loss = adversarial_g_loss + content_loss + gram_loss + color_loss
            g_loss.backward()
            optimizer_g.step()

            # adversarial_g_loss_epoch.append(adversarial_g_loss_epoch)
            # content_loss_epoch.append(content_loss)
            # gram_loss_epoch.append(gram_loss)
            # color_loss_epoch.append(color_loss_epoch)
            g_loss_epoch.append(g_loss)
            
            
        
        if (epoch+1) % train_config.save_epoch == 0:
            save_model(model_name="G_{}_at_epoch_{}".format(train_config.anime_dataset, epoch+1), model_dir=train_config.model_dir, model=G, optimizer=optimizer_g, epoch=epoch+1)
            save_model(model_name="D_{}_at_epoch_{}".format(train_config.anime_dataset, epoch+1), model_dir=train_config.model_dir, model=D, optimizer=optimizer_d, epoch=epoch+1)
            save_samples(train_config.save_img_dir, G, data_loader, train_config.batch_size, subname=train_config.anime_dataset+"_style")
            
        _d_loss = sum(d_loss_epoch)/len(d_loss_epoch)
        _g_loss = sum(g_loss_epoch)/len(g_loss_epoch)
        print("Epoch : {}/{}, D_LOSS : {}, G_LOSS: {}".format(epoch+1, train_config.total_epochs, _d_loss, _g_loss))
        
        
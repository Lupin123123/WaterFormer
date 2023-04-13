import torch
import torch.nn as nn
import torchvision
import torch.optim
import os
from torch.utils.data import DataLoader
from DataLoader.pair.dataloader import dehazing_loader
from model.WaterFormer_CSM import SwinTransformer

lr = 1e-3
num_epochs = 10
weight_decay = 1e-4
train_batch_size = 1
val_batch_size = 1
num_workers = 4
grad_clip_norm = 0.1
display_iter = 1
snapshot_iter = 1

ori_path = r'E:\workspace\work2\UIEB\train\GT'
hazy_path = r'E:\workspace\work2\UIEB\train\hazy'

snapshots_folder = './snapshots_folder'
sample_output_folder = './sample_output_folder'

dehaze_net = SwinTransformer(
    in_chans=3,
    patch_size=4,
    window_size=7,
    embed_dim=96,
    depths=(4, 4, 4, 4),
    num_heads=(8, 8, 8, 8)
)


optimizer = torch.optim.Adam(
    dehaze_net.parameters(),
    lr=lr,
    weight_decay=weight_decay
)

if __name__ == "__main__":

    ################################
    # 下面几句话要放在main中

    train_dataset = dehazing_loader(
        orig_images_path=ori_path,
        hazy_images_path=hazy_path
    )

    val_dataset = dehazing_loader(
        orig_images_path=ori_path,
        hazy_images_path=hazy_path,
        mode='val'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    ####################################
    
    dehaze_net.train()

    for epoch in range(num_epochs):
        for iteration, img_train in enumerate(train_loader):

            ori, hazy = img_train
            enhanced = dehaze_net(hazy)

            loss = torch.mean(torch.sum(torch.abs(enhanced - ori)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ((iteration + 1) % display_iter) == 0:
                print("Loss at iteration", iteration+1, ":", loss.item())

        # Validation Stage
        dehaze_net.eval()
        for iteration, img_val in enumerate(val_loader):
            ori, hazy = img_val
            enhanced = dehaze_net(hazy)
            save_path = os.path.join(
                sample_output_folder, str(iteration + 1) + ".jpg")
            cat_image = torch.cat((enhanced, ori), 0)
            cat_image = torch.cat((cat_image, hazy), 0)
            torchvision.utils.save_image(cat_image, save_path)

        state_dic_path = os.path.join(snapshots_folder, "dehazer.pth")
        torch.save(dehaze_net.state_dict(), state_dic_path)

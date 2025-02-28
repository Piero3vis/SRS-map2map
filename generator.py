# This script is used to modify the pretrained network from the paper
# The original network is used for 3D super resolution 
# It is trained on Ng_lr=64, Boxsize=100 Mpc/h for SR 512x particles
# Our strategy is 1) load the pretrained model 
# 2) freeze earlier layers
# 3) change the projection layers

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from map2map.models.srsgan import G


# # Load the checkpoint
# checkpoint = torch.load("SRmodel/G_z0.pt", map_location="cpu")
# pretrained_model = checkpoint['model']

# original_model = G(in_chan=1, out_chan=1, scale_factor=16, chan_base=128, chan_min=64, chan_max=128)
# original_model.load_state_dict(pretrained_model)

import torch
import torch.nn as nn
from map2map.models.srsgan import *
# Assuming your original model class is defined as follows:
from math import log2



# ------------------------------------------------------------------------------
# Define a modified version that only uses the first block (for a 2× upscale)
class ModifiedG(nn.Module):
    def __init__(self, original_model):
        super(ModifiedG, self).__init__()
        # Reuse the initial convolution and activation from the original model
        self.block0 = original_model.block0
        
        # Instead of using all upscaling blocks, keep only the first block.
        # This means only one doubling of the spatial resolution (64 -> 128 per dim).
        self.blocks = nn.ModuleList([original_model.blocks[0]])
        
        # If needed, you can also modify the projection layer in this block.
        # For example:
        # self.blocks[0].proj[0] = nn.Conv3d(
        #     in_channels=self.blocks[0].proj[0].in_channels,
        #     out_channels=self.blocks[0].proj[0].out_channels,
        #     kernel_size=1, stride=1, padding=0
        # )
        # (The pretrained weights could be reinitialized or kept as-is.)

    def forward(self, x):
        y = x
        x = self.block0(x)
        for block in self.blocks:
            x, y = block(x, y)
        return y

# ------------------------------------------------------------------------------
# Load the pretrained checkpoint
# (Your checkpoint contains keys 'epoch' and 'model'; we extract the state_dict.)
checkpoint = torch.load("SRmodel/G_z0.pt", map_location="cpu")
pretrained_state_dict = checkpoint['model']

# Instantiate the original model.
# Note: Based on your checkpoint state dict (which contains 3 blocks), the original
# model was likely created with scale_factor=8 (since round(log2(8)) == 3).
in_chan = 6
out_chan = 6
scale_factor = 8  # Adjust to match the training configuration of the checkpoint
original_model = G(in_chan, out_chan, scale_factor=scale_factor,
                   chan_base=512, chan_min=64, chan_max=512, cat_noise=False)

# Load the pretrained weights into the original model.
original_model.load_state_dict(pretrained_state_dict)

# Create the modified model that only uses the first upscaling block.
modified_model = ModifiedG(original_model)

# Optionally, freeze the reused weights to only train new components (if any)
# for param in modified_model.block0.parameters():
#     param.requires_grad = False
# for param in modified_model.blocks.parameters():
#     for p in param.parameters():
#         p.requires_grad = False

# Now, modified_model will take an input of shape (6, 64, 64, 64) and produce
# an output with spatial dimensions 128×128×128 (2× upscale per dim).
import torch

# Define your epoch number (adjust as needed)
epoch = 0  # or any other epoch value

# Create a checkpoint dictionary
checkpoint = {
    "epoch": epoch,
    "model": modified_model.state_dict()  # Save the state dict of your modified model
}

# Save the checkpoint to a .pt file
torch.save(checkpoint, "SRmodel/G_z0_modified.pt")

print("Checkpoint saved as 'SRmodel/G_z0_modified.pt'")


# checkpoint = {
#     'epoch': modified_model.state_dict['epoch'],
#     'model': modified_model.state_dict()['model']    
# }
# torch.save(checkpoint, 'SRmodel/G_z0_modified.pt')


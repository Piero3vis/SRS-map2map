# This script is used to modify the pretrained network from the paper
# The original network is used for 3D super resolution 
# It is trained on Ng_lr=64, Boxsize=100 Mpc/h for SR 512x particles
# Our strategy is 1) load the pretrained model 
# 2) freeze earlier layers
# 3) change the projection layers


import torch
import torch.nn as nn
from map2map.models.srsgan import *
# Assuming your original model class is defined as follows:
from math import log2


# ------------------------------------------------------------------------------
# Define a modified version that uses a specified number of blocks
class ModifiedG(nn.Module):
    def __init__(self, original_model, num_blocks=1):
        """
        Args:
            original_model: The pretrained model
            num_blocks: Number of blocks to keep (1 for 2×, 2 for 4×, 3 for 8×)
        """
        super(ModifiedG, self).__init__()
        # Reuse the initial convolution and activation from the original model
        self.block0 = original_model.block0
        
        # Keep specified number of blocks
        self.blocks = nn.ModuleList(
            original_model.blocks[:num_blocks]
        )
        
        # Store upsampling factor for reference
        self.scale_factor = 2 ** num_blocks

    def forward(self, x):
        y = x
        x = self.block0(x)
        for block in self.blocks:
            x, y = block(x, y)
        return y

# ------------------------------------------------------------------------------
# Load and modify the model
checkpoint = torch.load("SRmodel/G_z0.pt", map_location="cpu")
pretrained_state_dict = checkpoint['model']

# Original model setup
in_chan = 6
out_chan = 6
scale_factor = 8
original_model = G(in_chan, out_chan, scale_factor=scale_factor,
                  chan_base=512, chan_min=64, chan_max=512, cat_noise=False)

# Load pretrained weights
original_model.load_state_dict(pretrained_state_dict)

# Create modified model with desired number of blocks
num_blocks = 2  # Use 2 blocks for 4× upsampling
modified_model = ModifiedG(original_model, num_blocks=num_blocks)
print(f"Modified model will perform {2**num_blocks}× upsampling")

# Save the modified model
checkpoint = {
    "epoch": 0,
    "model": modified_model.state_dict()
}
torch.save(checkpoint, f"SRmodel/G_z0_modified_{2**num_blocks}x.pt")


import numpy as np
import plotly.graph_objects as go

# Load data
file_path = "./data/PART_010.npy"
data = np.load(file_path)

# Extract displacement vectors (first 3 channels)
x_disp, y_disp, z_disp = data[0], data[1], data[2]

# Define base positions (normalized grid indices)
grid_size = x_disp.shape[0]  # Should be 64
box_size = 100000  # Target box size
scale_factor = box_size / grid_size  # Scaling from 64³ to 100,000³

X, Y, Z = np.mgrid[:grid_size, :grid_size, :grid_size] * scale_factor

# Compute displaced positions in physical space
X_disp = X + x_disp * scale_factor
Y_disp = Y + y_disp * scale_factor
Z_disp = Z + z_disp * scale_factor

# Flatten for scatter plot
X_disp, Y_disp, Z_disp = X_disp.flatten(), Y_disp.flatten(), Z_disp.flatten()

# Create 3D scatter plot with blue particles
fig = go.Figure(data=go.Scatter3d(
    x=X_disp, y=Y_disp, z=Z_disp,
    mode="markers",
    marker=dict(size=1, opacity=0.3, color="blue")  # Blue color for filaments
))

fig.update_layout(title="3D Cosmic Structure: Filaments & Voids",
                  scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))

fig.show()


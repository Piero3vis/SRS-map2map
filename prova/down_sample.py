import numpy as np
import os, argparse

def load_lr_data(file_path):
    """Load LR data from either BigFile or .npy format."""
    if file_path.endswith('.npy'):
        print(f'loading {file_path} from .npy file')
        lr_data = np.load(file_path)
        if lr_data.shape[0] < 3:
            raise ValueError("The .npy file must have at least 3 channels for positions.")
    return lr_data

def down_sample(lr_data, downsample_factor):
    print(f'shape of lr_data: {lr_data.shape}')
    
    # Ensure downsample_factor is valid
    if downsample_factor <= 0:
        raise ValueError("downsample_factor must be greater than 0")
    
    # Calculate the number of samples to take
    num_samples = (lr_data.shape[1] // downsample_factor) * (lr_data.shape[2] // downsample_factor) * (lr_data.shape[3] // downsample_factor)
    
    # Randomly select indices for downsampling
    random_indices = np.random.choice(lr_data.shape[1] * lr_data.shape[2] * lr_data.shape[3], num_samples, replace=False)
    
    # Initialize downsampled data
    downsampled_data = np.zeros((lr_data.shape[0], lr_data.shape[1] // downsample_factor, lr_data.shape[2] // downsample_factor, lr_data.shape[3] // downsample_factor))
    
    # Fill downsampled data with random samples
    for i in range(num_samples):
        # Calculate the original indices from the random index
        original_index = random_indices[i]
        z = original_index // (lr_data.shape[2] * lr_data.shape[3])
        y = (original_index // lr_data.shape[3]) % lr_data.shape[2]
        x = original_index % lr_data.shape[3]
        
        # Calculate the new indices
        new_z = z // downsample_factor
        new_y = y // downsample_factor
        new_x = x // downsample_factor
        
        # Assign the value to the downsampled data
        downsampled_data[:, new_z, new_y, new_x] = lr_data[:, z, y, x]
    
    return downsampled_data

def check_shape(lr_data,  ds_data, downsample_factor):
    """Check if the shape of the downsampled data is correct."""
    
    # Ensure downsample_factor is valid
    if downsample_factor <= 0:
        raise ValueError("downsample_factor must be greater than 0")
    
    # Check if the original dimensions are divisible by the downsample_factor
    if lr_data.shape[1] % downsample_factor != 0 or lr_data.shape[2] % downsample_factor != 0 or lr_data.shape[3] % downsample_factor != 0:
        raise ValueError("The dimensions of lr_data must be divisible by downsample_factor.")
    
    expected_shape = (lr_data.shape[0], 
                      lr_data.shape[1] // downsample_factor, 
                      lr_data.shape[2] // downsample_factor, 
                      lr_data.shape[3] // downsample_factor)
    
    print(f'Expected shape: {expected_shape}, Actual shape: {ds_data.shape}')
    return ds_data.shape == expected_shape

def save_downsampled_data(downsampled_data, output_path, downsample_factor):
    """Save downsampled data to a .npy file."""
    print(f'saving downsampled data to {output_path+f"_downsampled_{downsample_factor}.npy"}')
    np.save(output_path+f'_downsampled_{downsample_factor}.npy', downsampled_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Downsample LR data')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the LR data file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the downsampled data')
    parser.add_argument('--downsample_factor', type=int, required=True, help='Downsample factor')
    args = parser.parse_args()

    lr_data = load_lr_data(args.input_path)
    downsampled_data = down_sample(lr_data, args.downsample_factor)
    if check_shape(lr_data, downsampled_data, args.downsample_factor):
        print(f'shape of downsampled data is correct: {downsampled_data.shape}')
        save_downsampled_data(downsampled_data, args.output_path, args.downsample_factor)
    else:
        print(f'shape of downsampled data is not correct: {downsampled_data.shape}')


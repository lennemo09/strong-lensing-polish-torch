

import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F

import matplotlib.pylab as plt


class WDSR(nn.Module):
    def __init__(self, num_residual_blocks=32, num_features=32, scale_factor=2):
        super(WDSR, self).__init__()
        self.scale_factor = scale_factor
        
        # Initial convolution
        self.conv_first = nn.Conv2d(1, num_features, kernel_size=3, padding=1)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            WDSRBlock(num_features) for _ in range(num_residual_blocks)
        ])
        
        # Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor)
        )
        
        # Final convolution
        self.conv_last = nn.Conv2d(num_features, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_first(x)
        residual = x
        for block in self.residual_blocks:
            x = block(x)
        x += residual
        x = self.upsample(x)
        x = self.conv_last(x)
        return x

class WDSRBlock(nn.Module):
    def __init__(self, num_features):  # Use double underscores for __init__
        super(WDSRBlock, self).__init__()  # Use double underscores for __init__
        self.conv1 = nn.Conv2d(num_features, num_features * 4, stride=1, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features * 4, num_features, stride=1, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x += residual
        return x

class WDSRpsf(nn.Module):
    def __init__(self, num_residual_blocks=32, num_features=32, scale_factor=2):
        super(WDSRpsf, self).__init__()
        self.scale_factor = scale_factor
        
        # Initial convolution (now accepts 2 channels: image and PSF)
        self.conv_first = nn.Conv2d(2, num_features, kernel_size=3, padding=1)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            WDSRBlockpsf(num_features) for _ in range(num_residual_blocks)
        ])
        
        # Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale_factor ** 2), 
            kernel_size=3, padding=1, stride=1),
            nn.PixelShuffle(scale_factor)
        )
        
        # Final convolution
        self.conv_last = nn.Conv2d(num_features, 1, kernel_size=3, stride=1, padding=1)
        
        # Remove the psf_conv layer as we'll apply the PSF differently
        
    def forward(self, x, psf):
        # Resize PSF to match input image dimensions
        psf_resized = F.interpolate(psf, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Combine input image and resized PSF
        x = torch.cat([x, psf_resized], dim=1)
        
        x = self.conv_first(x)
        residual = x
        for block in self.residual_blocks:
            x = block(x)
        x += residual
        x = self.upsample(x)
        x = self.conv_last(x)
        
        # Apply PSF convolution
        # Ensure PSF is the right size for convolution
        if psf.shape[2:] != (3, 3):
            psf = F.interpolate(psf, size=(3, 3), mode='bilinear', align_corners=False)
        
        # Apply PSF convolution manually for each item in the batch
        batch_size = x.shape[0]
        output = []
        for i in range(batch_size):
            current_psf = psf[i]
            output.append(F.conv2d(x[i].unsqueeze(0), current_psf.unsqueeze(0), padding=1, stride=1))
        
        x = torch.cat(output, dim=0)
        
        return x

class WDSRBlockpsf(nn.Module):
    def __init__(self, num_features):  # Use double underscores for __init__
        super(WDSRBlockpsf, self).__init__()  # Use double underscores for __init__
        self.conv1 = nn.Conv2d(num_features, num_features * 4, stride=1, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features * 4, num_features, stride=1, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x += residual
        return x

class SuperResolutionDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, start_num, end_num,
                 crop_size=256, transform=None, scale_factor=2):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.transform = transform
        self.image_files = [f"{i:04d}.npy" for i in range(start_num, end_num + 1)]
        self.crop_size = crop_size
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        #hr_image = Image.open(os.path.join(self.hr_dir, img_name))
        #lr_image = Image.open(os.path.join(self.lr_dir, img_name.replace('.png', 'x%d.png' % self.scale_factor)))

        hr_image = np.load(os.path.join(self.hr_dir, img_name))

        if self.scale_factor != 1:
            lr_image = np.load(os.path.join(self.lr_dir, img_name.replace('.npy', 'x%d.npy' % self.scale_factor)))
        else:
            lr_image = np.load(os.path.join(self.lr_dir, img_name))

        hr_image = torch.from_numpy(hr_image).squeeze()
        lr_image = torch.from_numpy(lr_image).squeeze()
        
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(hr_image, output_size=(self.crop_size, self.crop_size))
        hr_image = TF.crop(hr_image, i, j, h, w)
        lr_image = TF.crop(lr_image, i // self.scale_factor, j // self.scale_factor,
                           h // self.scale_factor, w // self.scale_factor)  # Adjust for LR size
        
        # Convert to numpy array and normalize
        hr_image = np.array(hr_image).astype(np.float32) #/ float(hr_image.max())  # Normalize 16-bit to [0, 1]
        lr_image = np.array(lr_image).astype(np.float32) #/ float(lr_image.max())
        
        # Convert to tensor
        hr_image = torch.from_numpy(hr_image).unsqueeze(0)
        lr_image = torch.from_numpy(lr_image).unsqueeze(0)
        
        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        return lr_image, hr_image

def train(model, train_loader, criterion, optimizer, device, psf=None):
    model.train()
    running_loss = 0.0
    for lr_imgs, hr_imgs in train_loader:
        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
        
        optimizer.zero_grad()
        if psf is None:
            outputs = model(lr_imgs)
        else:
            outputs = model(lr_imgs, psf)
        #print(lr_imgs)
        #print(outputs)
        loss = criterion(outputs, hr_imgs)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(train_loader)

# Validation function
def validate(model, val_loader, criterion, device, psf=None):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for lr_imgs, hr_imgs in val_loader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            if psf is None:
                outputs = model(lr_imgs)
            else:
                outputs = model(lr_imgs, psf)

            loss = criterion(outputs, hr_imgs)
            running_loss += loss.item()
    
    return running_loss / len(val_loader)

def validate_with_psnr(model, val_loader, criterion, device, psf=None):
    model.eval()
    total_loss = 0
    total_psnr = 0
    with torch.no_grad():
        for lr_imgs, hr_imgs in val_loader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            if psf is None:
                sr_imgs = model(lr_imgs)
            else:
                sr_imgs = model(lr_imgs, psf)
            loss = criterion(sr_imgs, hr_imgs)
            total_loss += loss.item()
            
            # Calculate PSNR
            for sr, hr in zip(sr_imgs, hr_imgs):
                sr_np = sr.cpu().numpy().transpose(1, 2, 0)  # CHW to HWC
                hr_np = hr.cpu().numpy().transpose(1, 2, 0)  # CHW to HWC
                total_psnr += calculate_psnr(sr_np, hr_np)
    
    avg_loss = total_loss / len(val_loader)
    avg_psnr = total_psnr / (len(val_loader) * val_loader.batch_size)
    return avg_loss, avg_psnr

# Assuming you have a PSNR calculation function. If not, I'll provide one.
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr  

def save_validation_results(model, val_loader, device, psf, output_dir, num_samples=5):
    """
    Save validation results, including input images, predicted images, true images, and PSNR.
    inputs:
        model (nn.Module): Trained model.
        val_loader (DataLoader): Validation DataLoader.
        device (torch.device): Device to run the model on.
        psf (torch.Tensor): PSF tensor.
        output_dir (str): Directory to save the plots.
        num_samples (int): Number of samples to save in the plot.
    """
    model.eval()
    samples = []
    with torch.no_grad():
        for lr_imgs, hr_imgs in val_loader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            if psf is None:
                sr_imgs = model(lr_imgs)
            else:
                sr_imgs = model(lr_imgs, psf)
            
            for i in range(len(lr_imgs)):
                psnr = calculate_psnr(
                    sr_imgs[i].cpu().numpy(), hr_imgs[i].cpu().numpy()
                )
                samples.append((lr_imgs[i].cpu(), sr_imgs[i].cpu(), hr_imgs[i].cpu(), psnr))
                if len(samples) >= num_samples:
                    break
            if len(samples) >= num_samples:
                break

    # Create the plot
    fig, axes = plt.subplots(num_samples, 4, figsize=(15, 3 * num_samples))
    for idx, (lr_img, sr_img, hr_img, psnr) in enumerate(samples):
        axes[idx, 0].imshow(lr_img.squeeze(), cmap='gray')
        axes[idx, 0].set_title(f"Input ({lr_img.squeeze().shape})")
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(sr_img.squeeze(), cmap='gray')
        axes[idx, 1].set_title(f"Predicted ({sr_img.squeeze().shape})")
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(hr_img.squeeze(), cmap='gray')
        axes[idx, 2].set_title("True")
        axes[idx, 2].axis('off')

        axes[idx, 3].text(0.5, 0.5, f"PSNR: {psnr:.2f} dB", ha='center', va='center', fontsize=12)
        axes[idx, 3].axis('off')

    plt.tight_layout()
    results_path = os.path.join(output_dir, "validation_results.png")
    plt.savefig(results_path)
    plt.close()
    print(f"Validation results saved to {results_path}")


def save_training_plot(train_losses, val_losses, val_psnr, save_path):
    """
    Save a plot showing training and validation MSE losses and validation PSNR as separate subplots.
    inputs:
        train_losses (list): List of training MSE losses.
        val_losses (list): List of validation MSE losses.
        val_psnr (list): List of validation PSNR scores.
        save_path (str): File path to save the plot.
    """
    epochs = range(1, len(train_losses) + 1)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Training Loss
    axes[0, 0].plot(epochs, train_losses, label="Train Loss (MSE)", marker='o', linestyle='-')
    axes[0, 0].set_title("Training Loss (MSE)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True)

    # Validation Loss
    axes[0, 1].plot(epochs, val_losses, label="Validation Loss (MSE)", marker='o', linestyle='-')
    axes[0, 1].set_title("Validation Loss (MSE)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True)

    # Validation PSNR
    axes[1, 0].plot(epochs, val_psnr, label="Validation PSNR", marker='o', linestyle='--')
    axes[1, 0].set_title("Validation PSNR")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("PSNR")
    axes[1, 0].grid(True)

    # Empty subplot (optional or for additional metrics)
    axes[1, 1].axis('off')  # Leave it empty or use it for additional plots/metrics

    # Adjust layout and save the plot
    fig.suptitle("Training and Validation Metrics", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the main title
    plt.savefig(save_path)
    plt.close()
    print(f"Training plot saved to {save_path}")


def get_available_memory(device_index):
    """
    Returns the available memory on the specified GPU in GB.
    """
    torch.cuda.empty_cache()  # Clear unused cached memory
    memory_stats = torch.cuda.memory_stats(device_index)
    allocated = memory_stats["allocated_bytes.all.current"] / (1024 ** 3)  # Convert to GB
    reserved = memory_stats["reserved_bytes.all.current"] / (1024 ** 3)    # Convert to GB
    total_memory = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)  # Convert to GB
    available_memory = total_memory - max(allocated, reserved)
    return available_memory


def main(datadir, scale=1, model_name=None, psf=False):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    num_epochs = 500
    batch_size = 4
    learning_rate = 0.0001

    output_dir = f'./runs/final_data/temp_run_strong/'
    os.makedirs(output_dir, exist_ok=True) 

    print("Output will be saved to:", output_dir)

    if psf:
        model = WDSRpsf(scale_factor=1).to(device)
        psfarr = np.load(f'{datadir}/psf/psf_ideal.npy')
        npsf = len(psfarr)
        psfarr = psfarr[npsf//2-256:npsf//2+256, npsf//2-256:npsf//2+256]
        psfarr = psfarr[None,None] * np.ones([batch_size,1,1,1])
        psfarr = torch.from_numpy(psfarr).to(device).float()

        psf_to_save = psfarr[0, 0].detach().cpu()  # Extract the 2D PSF and move it to CPU if needed
        psf_np = psf_to_save.numpy()  # Convert the tensor to a NumPy array

        # Normalize PSF values to 0-255 for saving as an 8-bit PNG
        psf_normalized = (psf_np - psf_np.min()) / (psf_np.max() - psf_np.min()) * 255
        psf_image = Image.fromarray(psf_normalized.astype('uint8'))  # Convert to 8-bit image
        psf_image.save(f'{output_dir}psf.png')  # Save as PNG

        psf_log = np.log10(psf_normalized)

        # Plot the log-scaled PSF
        plt.figure(figsize=(6, 6))
        plt.imshow(psf_log, cmap='viridis')
        plt.title('Log-Scaled PSF')
        plt.axis('off')  # Optional: turn off the axes

        # Save the figure
        plt.savefig(f'{output_dir}psf_log_scale.png', bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        model = WDSR(scale_factor=scale).to(device)
        psfarr = None

    if model_name != None:
        model.load_state_dict(torch.load(model_name))
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    best_val_loss = float('inf')
    best_val_psnr = 0.0
    train_losses = []
    val_losses = []
    val_psnr = []
    plot_save_path = output_dir + 'training_metrics.png'
    
    crop_size = 512 
    # Load datasets
    if scale != 1:
        train_dataset = SuperResolutionDataset('%s/POLISH_train_true/' % datadir, '%s/POLISH_train_dirty_lowres_x%d/' % (datadir,scale), 0, 799, scale_factor=scale, crop_size=crop_size)
        val_dataset = SuperResolutionDataset('%s/POLISH_valid_true/' % datadir, '%s/POLISH_valid_dirty_lowres_x%d/' % (datadir, scale), 800, 899, scale_factor=scale, crop_size=crop_size)
        print(f'Scale factor: {scale}')
        print(f"Loading train data from: {'%s/POLISH_train_true/' % datadir} and {'%s/POLISH_train_dirty_lowres_x%d/' % (datadir,scale)}")
        print(f"Loading validation data from: {'%s/POLISH_valid_true/' % datadir} and {'%s/POLISH_valid_dirty_lowres_x%d/' % (datadir, scale)}")
    else:
        train_dataset = SuperResolutionDataset('%s/POLISH_train_true/' % datadir, '%s/POLISH_train_dirty/' % (datadir), 0, 799, scale_factor=scale, crop_size=crop_size)
        val_dataset = SuperResolutionDataset('%s/POLISH_valid_true/' % datadir, '%s/POLISH_valid_dirty/' % (datadir), 800, 899, scale_factor=scale, crop_size=crop_size)
        print(f'Scale factor: {scale}')
        print(f"Loading train data from: {'%s/POLISH_train_true/' % datadir} and {'%s/POLISH_train_dirty/' % (datadir)}")
        print(f"Loading validation data from: {'%s/POLISH_valid_true/' % datadir} and {'%s/POLISH_valid_dirty/' % (datadir)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device, psf=psfarr)
        val_loss, avg_psnr = validate_with_psnr(model, val_loader, criterion, device, psf=psfarr)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_psnr.append(avg_psnr)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val PSNR: {avg_psnr:.2f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir + f'best_loss_model.pth')

        if avg_psnr > best_val_psnr:
            best_val_psnr = avg_psnr
            torch.save(model.state_dict(), output_dir + f'best_psnr_model.pth')

        # Save metrics plot every 5 epochs
        if (epoch) % 2 == 0:
            save_training_plot(train_losses, val_losses, val_psnr, plot_save_path)
            save_validation_results(model, val_loader, device, psf=psfarr, output_dir=output_dir)
            
    # Save the final model and plot
    torch.save(model.state_dict(), output_dir + f'final_model.pth')
    save_training_plot(train_losses, val_losses, val_psnr, plot_save_path)


if __name__=='__main__':
    try:
        model_name = sys.argv[3]
    except:
        model_name = None

    try:
        use_psf = int(sys.argv[3])
    except:
        use_psf = 0

    print(f'argv[1]: {sys.argv[1]}, scale: argv[2]: {int(sys.argv[2])}')
    main(sys.argv[1], int(sys.argv[2]), model_name=None, psf=(use_psf==1))
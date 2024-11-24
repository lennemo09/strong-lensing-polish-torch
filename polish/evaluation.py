import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr, mean_squared_error
from skimage.transform import resize
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

class WDSR(nn.Module):
    def __init__(self, num_residual_blocks=32, num_features=32, scale_factor=2):
        super(WDSR, self).__init__()
        self.scale_factor = scale_factor
        self.conv_first = nn.Conv2d(1, num_features, kernel_size=3, padding=1)
        self.residual_blocks = nn.ModuleList([
            WDSRBlock(num_features) for _ in range(num_residual_blocks)
        ])
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor)
        )

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
    def __init__(self, num_features):
        super(WDSRBlock, self).__init__()
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
                 crop_size=None, transform=None, scale_factor=2):
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
        hr_image_path = os.path.join(self.hr_dir, img_name)
        lr_image_path = os.path.join(self.lr_dir, img_name)

        hr_image = np.load(hr_image_path)
        lr_image = np.load(lr_image_path)
        print(f"Loaded HR image '{img_name}': min={hr_image.min()}, max={hr_image.max()}, shape={hr_image.shape}")
        print(f"Loaded LR image '{img_name}': min={lr_image.min()}, max={lr_image.max()}, shape={lr_image.shape}")

        hr_image = torch.from_numpy(hr_image).squeeze()
        lr_image = torch.from_numpy(lr_image).squeeze()

        if self.crop_size is not None:
            i, j, h, w = transforms.RandomCrop.get_params(hr_image, output_size=(self.crop_size, self.crop_size))
            hr_image = TF.crop(hr_image, i, j, h, w)
            lr_image = TF.crop(lr_image, i // self.scale_factor, j // self.scale_factor,
                               h // self.scale_factor, w // self.scale_factor)
        else:
            
            pass

        
        hr_image = np.array(hr_image).astype(np.float32)
        lr_image = np.array(lr_image).astype(np.float32)

        
        hr_image = torch.from_numpy(hr_image).unsqueeze(0)
        lr_image = torch.from_numpy(lr_image).unsqueeze(0)

        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        return lr_image, hr_image, img_name

class ResidualLearningDataset(Dataset):
    def __init__(self, hr_dir, residual_dr, lr_dir, start_num, end_num, nbit=0,
                 crop_size=None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.nbit = nbit
        self.residual_dir = residual_dr
        self.image_files = [f"{i:04d}.npy" for i in range(start_num, end_num + 1)]
        self.crop_size = crop_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        hr_image_path = os.path.join(self.hr_dir, img_name)
        lr_image_path = os.path.join(self.lr_dir, img_name)
        residual_image_path = os.path.join(self.residual_dir, img_name)

        hr_image = np.load(hr_image_path)
        lr_image = np.load(lr_image_path)
        residual_image = np.load(residual_image_path)

        
        print(f"Loaded HR image '{img_name}': min={hr_image.min()}, max={hr_image.max()}, shape={hr_image.shape}")
        print(f"Loaded LR image '{img_name}': min={lr_image.min()}, max={lr_image.max()}, shape={lr_image.shape}")
        print(f"Loaded residual image '{img_name}': min={residual_image.min()}, max={residual_image.max()}, shape={residual_image.shape}")

        if self.nbit > 0:
            max_value = (2**(self.nbit//2)-1)
            hr_image = hr_image * max_value
            lr_image = lr_image * max_value
            residual_image = residual_image * max_value

        hr_image = torch.from_numpy(hr_image).squeeze()
        lr_image = torch.from_numpy(lr_image).squeeze()
        residual_image = torch.from_numpy(residual_image).squeeze()

        if self.crop_size is not None:
            
            i, j, h, w = transforms.RandomCrop.get_params(hr_image, output_size=(self.crop_size, self.crop_size))
            hr_image = TF.crop(hr_image, i, j, h, w)
            lr_image = TF.crop(lr_image, i, j, h, w)
            residual_image = TF.crop(residual_image, i, j, h, w)
        else:
            
            pass

        hr_image = np.array(hr_image).astype(np.float32)
        lr_image = np.array(lr_image).astype(np.float32)
        residual_image = np.array(residual_image).astype(np.float32)
        
        
        hr_image = torch.from_numpy(hr_image).unsqueeze(0)
        lr_image = torch.from_numpy(lr_image).unsqueeze(0)
        residual_image = torch.from_numpy(residual_image).unsqueeze(0)

        return lr_image, hr_image, residual_image, img_name



def normalize_image(img, max_value=None):
    """
    Normalize image to [0,1] range
    """
    if max_value is not None:
        return img / max_value
    else:
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        else:
            return img - img_min  

def compute_metrics(output, target):
    """
    Compute SSIM, MSE, PSNR between output and target images
    Both inputs should be numpy arrays in [0,1] range
    """
    ssim_value = ssim(target, output, data_range=1.0)
    mse_value = mean_squared_error(target, output)
    psnr_value = psnr(target, output, data_range=1.0)
    return ssim_value, mse_value, psnr_value

def save_comparison_plots(image_indices, metric_name, method_name, output_dir, images_dict, img_names_list, metrics_dict):
    for idx in image_indices:
        img_name = img_names_list[idx]
        dirty_img = images_dict['dirty'][idx]
        hr_img = images_dict['hr'][idx]

        if method_name.startswith('strong'):
            
            output_img = images_dict['strong_output'][idx]
            residual_img = images_dict['strong_residual'][idx]

            images = [dirty_img, output_img, hr_img, residual_img]
            min_max_values = [
                f"Min: {dirty_img.min():.3f}, Max: {dirty_img.max():.3f}",
                f"Min: {output_img.min():.3f}, Max: {output_img.max():.3f}",
                f"Min: {hr_img.min():.3f}, Max: {hr_img.max():.3f}",
                f"Min: {residual_img.min():.3f}, Max: {residual_img.max():.3f}"
            ]
            titles = [
                f'Dirty Image\n{min_max_values[0]}',
                f'Strong Model Output\nSSIM: {metrics_dict["strong"][idx]["SSIM"]:.3f}, MSE: {metrics_dict["strong"][idx]["MSE"]:.3f}, PSNR: {metrics_dict["strong"][idx]["PSNR"]:.3f}\n{min_max_values[1]}',
                f'Ground Truth\n{min_max_values[2]}',
                f'Residual (True - Output)\n{min_max_values[3]}'
            ]
        elif method_name.startswith('residual'):
            
            gen_residual_img = images_dict['residual_generated'][idx]
            reconstructed_img = images_dict['residual_output'][idx]
            residual_img = images_dict['residual_residual'][idx]

            images = [dirty_img, gen_residual_img, reconstructed_img, hr_img, residual_img]
            min_max_values = [
                f"Min: {dirty_img.min():.3f}, Max: {dirty_img.max():.3f}",
                f"Min: {gen_residual_img.min():.3f}, Max: {gen_residual_img.max():.3f}",
                f"Min: {reconstructed_img.min():.3f}, Max: {reconstructed_img.max():.3f}",
                f"Min: {hr_img.min():.3f}, Max: {hr_img.max():.3f}",
                f"Min: {residual_img.min():.3f}, Max: {residual_img.max():.3f}"
            ]
            titles = [
                f'Dirty Image\n{min_max_values[0]}',
                f'Generated Residual\n{min_max_values[1]}',
                f'Reconstructed Image\nSSIM: {metrics_dict["residual"][idx]["SSIM"]:.3f}, MSE: {metrics_dict["residual"][idx]["MSE"]:.3f}, PSNR: {metrics_dict["residual"][idx]["PSNR"]:.3f}\n{min_max_values[2]}',
                f'Ground Truth\n{min_max_values[3]}',
                f'Residual (True - Reconstructed)\n{min_max_values[4]}'
            ]
        else:
            continue  

        
        num_images = len(images)
        fig, axes = plt.subplots(2, num_images, figsize=(5*num_images, 10))

        
        for ax, img, title in zip(axes[0], images, titles):
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')

        
        for ax, img in zip(axes[1], images):
            
            ax.imshow(np.log10(np.abs(img) + 1e-8), cmap='gray')  
            ax.axis('off')

        plt.suptitle(f'Image: {img_name}, Metric: {metric_name}, Method: {method_name}')
        plot_filename = f'comparison_{metric_name}_{method_name}_{img_name}.png'
        plt.savefig(os.path.join(output_dir, plot_filename))
        plt.close()

def main():
    
    strong_model_path = '/u8/d/len/code/strong-lensing-polish-torch/runs/final_data/run3_cropsize512_s2/final_model.pth'
    residual_model_path = '/u8/d/len/code/strong-lensing-polish-torch/runs/final_data/run7_residual_IvsI_cropsize512/final_model.pth'

    
    data_dir = '/scratch/ondemand28/len/data/DSA_PSF_1024_x2_stronglens'

    
    nbit = 16  

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    strong_model = WDSR(scale_factor=2).to(device)
    strong_model.load_state_dict(torch.load(strong_model_path, map_location=device))

    residual_model = WDSR(scale_factor=1).to(device)
    residual_model.load_state_dict(torch.load(residual_model_path, map_location=device))

    
    strong_model.eval()
    residual_model.eval()

    
    crop_size = None  
    start_idx = 800
    end_idx = 899

    
    strong_val_dataset = SuperResolutionDataset(
        os.path.join(data_dir, 'POLISH_valid_true'),
        os.path.join(data_dir, 'POLISH_valid_dirty'),
        start_idx, end_idx, scale_factor=2, crop_size=crop_size
    )

    
    residual_val_dataset = ResidualLearningDataset(
        os.path.join(data_dir, 'POLISH_valid_true'),
        os.path.join(data_dir, 'POLISH_valid_residual'),
        os.path.join(data_dir, 'POLISH_valid_dirty'),
        start_idx, end_idx, nbit=nbit, crop_size=crop_size
    )

    
    strong_val_loader = DataLoader(strong_val_dataset, batch_size=1, shuffle=False)
    residual_val_loader = DataLoader(residual_val_dataset, batch_size=1, shuffle=False)

    
    dirty_imgs_list = []
    hr_imgs_list = []
    img_names_list = []

    
    outputs_strong_list = []
    residuals_strong_list = []
    metrics_strong_list = []

    
    generated_residuals_list = []
    reconstructed_residuals_list = []
    residuals_residual_list = []
    metrics_residual_list = []

    
    for idx, ((lr_img_strong, hr_img_strong, img_name_strong), (lr_img_residual, hr_img_residual, res_img, img_name_residual)) in enumerate(zip(strong_val_loader, residual_val_loader)):
        
        img_name = img_name_strong[0]
        assert img_name == img_name_residual[0], "Image names do not match!"

        
        lr_img_strong = lr_img_strong.to(device)
        hr_img_strong = hr_img_strong.to(device)

        lr_img_residual = lr_img_residual.to(device)
        hr_img_residual = hr_img_residual.to(device)
        res_img = res_img.to(device)

        
        print(f"Processing image '{img_name}':")
        print(f"  Strong model input LR image: min={lr_img_strong.min().item()}, max={lr_img_strong.max().item()}, shape={lr_img_strong.shape}")
        print(f"  Residual model input LR image: min={lr_img_residual.min().item()}, max={lr_img_residual.max().item()}, shape={lr_img_residual.shape}")

        
        with torch.no_grad():
            output_strong = strong_model(lr_img_strong)

        
        print(f"  Strong model output: min={output_strong.min().item()}, max={output_strong.max().item()}, shape={output_strong.shape}")

        
        output_strong_np = output_strong.cpu().numpy()[0, 0, :, :]

        
        with torch.no_grad():
            output_residual = residual_model(lr_img_residual)
            reconstructed_residual = lr_img_residual - output_residual  

        
        print(f"  Residual model output (generated residual): min={output_residual.min().item()}, max={output_residual.max().item()}, shape={output_residual.shape}")
        print(f"  Residual model reconstructed image: min={reconstructed_residual.min().item()}, max={reconstructed_residual.max().item()}, shape={reconstructed_residual.shape}")

        
        generated_residual_np = output_residual.cpu().numpy()[0, 0, :, :]
        reconstructed_residual_np = reconstructed_residual.cpu().numpy()[0, 0, :, :]

        
        hr_img = hr_img_residual.cpu().numpy()[0, 0, :, :]

        
        dirty_img_np = lr_img_residual.cpu().numpy()[0, 0, :, :]

        
        if output_strong_np.shape != hr_img.shape:
            output_strong_resized = resize(output_strong_np, hr_img.shape, mode='reflect', anti_aliasing=True)
        else:
            output_strong_resized = output_strong_np

        
        hr_img_normalized = normalize_image(hr_img)
        output_strong_normalized = normalize_image(output_strong_resized)

        
        print(f"Strong Model - True Image: min={hr_img_normalized.min()}, max={hr_img_normalized.max()}, shape={hr_img_normalized.shape}")
        print(f"Strong Model - Reconstructed Image: min={output_strong_normalized.min()}, max={output_strong_normalized.max()}, shape={output_strong_normalized.shape}")

        
        residual_strong = hr_img_normalized - output_strong_normalized

        
        ssim_strong, mse_strong, psnr_strong = compute_metrics(output_strong_normalized, hr_img_normalized)

        print(f"Strong SSIM: {ssim_strong}, MSE: {mse_strong}, PSNR: {psnr_strong}")

        
        outputs_strong_list.append(output_strong_normalized)
        residuals_strong_list.append(residual_strong)
        metrics_strong_list.append({'SSIM': ssim_strong, 'MSE': mse_strong, 'PSNR': psnr_strong})

        
        if reconstructed_residual_np.shape != hr_img.shape:
            reconstructed_residual_np = resize(reconstructed_residual_np, hr_img.shape, mode='reflect', anti_aliasing=True)

        
        hr_img_normalized_residual = normalize_image(hr_img)
        reconstructed_residual_normalized = normalize_image(reconstructed_residual_np)

        
        print(f"Residual Model - True Image: min={hr_img_normalized_residual.min()}, max={hr_img_normalized_residual.max()}, shape={hr_img_normalized_residual.shape}")
        print(f"Residual Model - Reconstructed Image: min={reconstructed_residual_normalized.min()}, max={reconstructed_residual_normalized.max()}, shape={reconstructed_residual_normalized.shape}")

        
        residual_residual = hr_img_normalized_residual - reconstructed_residual_normalized

        
        ssim_residual, mse_residual, psnr_residual = compute_metrics(reconstructed_residual_normalized, hr_img_normalized_residual)

        
        generated_residuals_list.append(normalize_image(generated_residual_np))
        reconstructed_residuals_list.append(reconstructed_residual_normalized)
        residuals_residual_list.append(residual_residual)
        metrics_residual_list.append({'SSIM': ssim_residual, 'MSE': mse_residual, 'PSNR': psnr_residual})

        
        dirty_imgs_list.append(dirty_img_np)
        hr_imgs_list.append(hr_img)
        img_names_list.append(img_name)

    
    metrics_list = []
    for idx in range(len(img_names_list)):
        metrics_list.append({
            'image_index': idx,
            'image_name': img_names_list[idx],
            'strong_SSIM': metrics_strong_list[idx]['SSIM'],
            'strong_MSE': metrics_strong_list[idx]['MSE'],
            'strong_PSNR': metrics_strong_list[idx]['PSNR'],
            'residual_SSIM': metrics_residual_list[idx]['SSIM'],
            'residual_MSE': metrics_residual_list[idx]['MSE'],
            'residual_PSNR': metrics_residual_list[idx]['PSNR']
        })

    metrics_df = pd.DataFrame(metrics_list)

    
    images_dict_strong = {
        'dirty': dirty_imgs_list,
        'hr': hr_imgs_list,
        'strong_output': outputs_strong_list,
        'strong_residual': residuals_strong_list,
    }

    images_dict_residual = {
        'dirty': dirty_imgs_list,
        'hr': hr_imgs_list,
        'residual_generated': generated_residuals_list,
        'residual_output': reconstructed_residuals_list,
        'residual_residual': residuals_residual_list,
    }

    output_dir = '/u8/d/len/code/strong-lensing-polish-torch/eval/'  
    os.makedirs(output_dir, exist_ok=True)

    metrics = ['SSIM', 'MSE', 'PSNR']
    top_results = {}

    for metric in metrics:
        
        
        if metric in ['SSIM', 'PSNR']:
            ascending = False
        else:
            ascending = True

        
        top_strong = metrics_df.sort_values(by=f'strong_{metric}', ascending=ascending).head(10)
        top_results[f'top_10_strong_{metric}'] = top_strong[['image_index', 'image_name', f'strong_{metric}']]

        
        image_indices = top_strong['image_index'].tolist()
        save_comparison_plots(image_indices, metric, 'strong_top10', output_dir, images_dict_strong, img_names_list, {'strong': metrics_strong_list})

        
        worst_strong = metrics_df.sort_values(by=f'strong_{metric}', ascending=not ascending).head(10)
        top_results[f'worst_10_strong_{metric}'] = worst_strong[['image_index', 'image_name', f'strong_{metric}']]

        
        image_indices = worst_strong['image_index'].tolist()
        save_comparison_plots(image_indices, metric, 'strong_worst10', output_dir, images_dict_strong, img_names_list, {'strong': metrics_strong_list})

        
        top_residual = metrics_df.sort_values(by=f'residual_{metric}', ascending=ascending).head(10)
        top_results[f'top_10_residual_{metric}'] = top_residual[['image_index', 'image_name', f'residual_{metric}']]

        
        image_indices = top_residual['image_index'].tolist()
        save_comparison_plots(image_indices, metric, 'residual_top10', output_dir, images_dict_residual, img_names_list, {'residual': metrics_residual_list})

        
        worst_residual = metrics_df.sort_values(by=f'residual_{metric}', ascending=not ascending).head(10)
        top_results[f'worst_10_residual_{metric}'] = worst_residual[['image_index', 'image_name', f'residual_{metric}']]

        
        image_indices = worst_residual['image_index'].tolist()
        save_comparison_plots(image_indices, metric, 'residual_worst10', output_dir, images_dict_residual, img_names_list, {'residual': metrics_residual_list})

        
        diff_metric = metrics_df[f'strong_{metric}'] - metrics_df[f'residual_{metric}']

        
        if metric in ['SSIM', 'PSNR']:
            
            top_diff_strong = diff_metric.sort_values(ascending=False).head(10)
            
            top_diff_residual = diff_metric.sort_values().head(10)
        else:
            
            
            top_diff_strong = diff_metric.sort_values().head(10)
            
            top_diff_residual = diff_metric.sort_values(ascending=False).head(10)

        
        top_diff_strong_indices = top_diff_strong.index
        top_diff_strong_df = metrics_df.loc[top_diff_strong_indices, ['image_index', 'image_name', f'strong_{metric}', f'residual_{metric}']]
        top_results[f'top_10_strong_beats_residual_{metric}'] = top_diff_strong_df

        
        top_diff_residual_indices = top_diff_residual.index
        top_diff_residual_df = metrics_df.loc[top_diff_residual_indices, ['image_index', 'image_name', f'strong_{metric}', f'residual_{metric}']]
        top_results[f'top_10_residual_beats_strong_{metric}'] = top_diff_residual_df

        

    
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)

    
    for key, df in top_results.items():
        df.to_csv(os.path.join(output_dir, f'{key}.csv'), index=False)

    print("Metrics and top results saved to", output_dir)

if __name__ == '__main__':
    main()

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm  
from ASDQE_model import DenoiseRatePredictor  
import pandas as pd
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset):
    def __init__(self, lg_dir, gt_dir, transform=None):
        """
        Initialize the dataset
        
        Parameters:
            lg_dir: Path to the low-quality image folder
            gt_dir: Path to the high-quality image folder
            transform: Image preprocessing transformation
        """
        self.lg_dir = lg_dir
        self.gt_dir = gt_dir
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.lg_files = sorted(os.listdir(lg_dir))
        self.gt_files = sorted(os.listdir(gt_dir))
        
        if len(self.lg_files) != len(self.gt_files):
            raise ValueError(f"The number of images in the lg folder ({len(self.lg_files)}) and the gt folder ({len(self.gt_files)}) do not match")
        
        for lg_file, gt_file in zip(self.lg_files, self.gt_files):
            if os.path.splitext(lg_file)[0] != os.path.splitext(gt_file)[0]:
                raise ValueError(f"File names do not match: {lg_file} vs {gt_file}")
    
    def __len__(self):
        return len(self.lg_files)
    
    def __getitem__(self, idx):
        """Get a pair of images (low quality and high quality)"""
        lg_img_path = os.path.join(self.lg_dir, self.lg_files[idx])
        gt_img_path = os.path.join(self.gt_dir, self.gt_files[idx])
        
        lg_image = Image.open(lg_img_path).convert('RGB')
        gt_image = Image.open(gt_img_path).convert('RGB')
        
        if self.transform:
            lg_image = self.transform(lg_image)
            gt_image = self.transform(gt_image)
        
        return {
            'lq': {'img': lg_image},
            'gt': {'hq': gt_image}
        }



# Load model
def load_model(model_path):
    """Load the trained model"""
    model = DenoiseRatePredictor().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print(f"Successfully loaded model: {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.eval()  
    return model

# Inference function
def infer(model, dataloader, method_name):
    """Inference on the dataset and collect results, add progress bar display"""
    predictions = []
    
    with torch.no_grad():  
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"推理 {method_name}")
        for i, batch in progress_bar:
            lq_images = batch['lq']['img'].to(device)  
            gt_images = batch['gt']['hq'].to(device)  
            
            outputs = model(lq_images, gt_images)
            
            pred_values = outputs.cpu().numpy().flatten()
            predictions.extend(pred_values)
            
            progress_bar.set_postfix({"Current batch": i+1, "Processed images": len(predictions)})
    
    return np.array(predictions)

# Statistical result function
def calculate_statistics(values):
    """Calculate and return statistical indicators"""
    stats = {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        '25%': np.percentile(values, 25),
        '50%': np.percentile(values, 50),
        '75%': np.percentile(values, 75),
        'max': np.max(values)
    }
    
    return stats

# Visualize comparison results
def visualize_comparison(all_stats, method_names):
    """Visualize the comparison results of different denoising methods"""
    stats_df = pd.DataFrame(all_stats)
    stats_df.index = method_names
    
    print("\n===== Statistical data comparison of different denoising methods =====")
    print(stats_df.T.to_string(float_format='%.6f'))
    stats_df.T.to_csv(
    'stats_transposed.csv', 
    float_format='%.6f',   
    index=True            
    )

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    base_lg_dir = "../Sample/WDD/origin" 
    denoise_dir = "../Sample/WDD/denoise/"

    denoising_methods = {
        "origin": base_lg_dir,  
        # "speckle@0.05":  denoise_dir + "speckle@0.05",
        # "speckle@0.1":  denoise_dir + "speckle@0.1",
        # "speckle@0.2":  denoise_dir + "speckle@0.2",
        # "gaussian":  denoise_dir + "gaussian",
        # "bm3d":  denoise_dir + "bm3d",
        # "bilateral":  denoise_dir + "bilateral",
        # "median": denoise_dir + "median",
        # "nlm":  denoise_dir + "nlm",
        "Teacher":  denoise_dir + "KDLAE-T",
        # "Student@0.02":  denoise_dir + "MultiFrameDenoiserStudentLite_30000",
        "Student@0.05":  denoise_dir + "KDLAE-S_prob@0.05",
        # "Student@0.08":  denoise_dir + "MultiFrameDenoiserStudentLite_66000",
        # "Student@0.11":  denoise_dir + "MultiFrameDenoiserStudentLite_78000",
        # "Student@0.15":  denoise_dir + "MultiFrameDenoiserStudentLite_90000",
        # "Student@0.20":  denoise_dir + "MultiFrameDenoiserStudentLite_100000",
        # "SimNFND": denoise_dir + "SimNFND2",
    }

    # base_lg_dir = "/mnt/mnt/data2/YTH/Dataset/CAMUS/CAMUS_dataset/test/ch4_resize"  
    # denoise_dir = '/mnt/mnt/data2/YTH/Dataset/CAMUS/CAMUS_dataset/denoise/'
    # denoising_methods = {
    #     "origin": base_lg_dir,  
    #     "speckle@0.05": denoise_dir + "speckle@0.05",
    #     "speckle@0.1": denoise_dir + "speckle@0.1",
    #     "speckle@0.2": denoise_dir + "speckle@0.2",
    #     "gaussian": denoise_dir + "gaussian",
    #     "bm3d": denoise_dir + "bm3d",
    #     "bilateral": denoise_dir + "bilateral",
    #     "median": denoise_dir + "median",
    #     "nlm": denoise_dir + "nlm",
    #     "Teacher": denoise_dir + "RestormerSuperResolutionParam2",
    #     "Student@0.02":  denoise_dir + "MultiFrameDenoiserStudentLite_30000",
    #     "Student@0.05":  denoise_dir + "MultiFrameDenoiserStudentLite_50000",
    #     "Student@0.08":  denoise_dir + "MultiFrameDenoiserStudentLite_66000",
    #     "Student@0.11":  denoise_dir + "MultiFrameDenoiserStudentLite_78000",
    #     "Student@0.15":  denoise_dir + "MultiFrameDenoiserStudentLite_90000",
    #     "Student@0.20":  denoise_dir + "MultiFrameDenoiserStudentLite_100000",
    #     "Restormer":denoise_dir + "Restormer",
    # }


    model_path = "weights/ASDQE.pth"  
    
    model = load_model(model_path)
    
    all_statistics = []
    method_names = list(denoising_methods.keys())
    
    print(f"Will evaluate {len(method_names)} denoising methods...")

    for method_name, gt_dir in denoising_methods.items():
        print(f"\n===== Processing denoising method: {method_name} =====")
        
        dataset = CustomImageDataset(base_lg_dir, gt_dir)
        dataloader = DataLoader(
            dataset,
            batch_size=1,  
            shuffle=False,  
            num_workers=0,  
            pin_memory=True
        )
        print(f"Data loading completed, {len(dataset)} image pairs")
        
        predictions = infer(model, dataloader, method_name)
        print(f"Inference completed, {len(predictions)} prediction values")
        
        stats = calculate_statistics(predictions)
        all_statistics.append(stats)
        
        print(f"\n===== {method_name} Inference result statistics =====")
        for key, value in stats.items():
            print(f"{key}: {value:.6f}")
    
    visualize_comparison(all_statistics, method_names)
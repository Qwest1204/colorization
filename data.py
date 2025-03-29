from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms as T
from pathlib import Path
import os
from PIL import Image

class ColorizeDataset(Dataset):
    def __init__(self, color_dir, bw_dir):
        self.color_dir = Path(color_dir)
        self.bw_dir = Path(bw_dir)

        # Проверка существования директорий
        if not self.color_dir.exists():
            raise FileNotFoundError(f"Color directory {color_dir} not found")
        if not self.bw_dir.exists():
            raise FileNotFoundError(f"B/W directory {bw_dir} not found")

        # Получаем список файлов с преобразованием Path в строку
        self.color_paths = [
            self.color_dir / fname 
            for fname in sorted(os.listdir(str(self.color_dir)))  # Исправлено: преобразование в строку
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        self.bw_paths = [
            self.bw_dir / fname 
            for fname in sorted(os.listdir(str(self.bw_dir)))  # Исправлено: преобразование в строку
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        # Проверки на пустоту
        if not self.color_paths:
            raise RuntimeError(f"No images found in {color_dir}")
        if not self.bw_paths:
            raise RuntimeError(f"No images found in {bw_dir}")
        if len(self.color_paths) != len(self.bw_paths):
            raise ValueError("Mismatch in number of color and BW images")

        self.transform = T.Compose([
            T.Resize(128),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.color_paths)
    
    def __getitem__(self, index):
        img_c = Image.open(self.color_paths[index]).convert('RGB')
        img_bw = Image.open(self.bw_paths[index]).convert('L')
        
        return self.transform(img_bw), self.transform(img_c)
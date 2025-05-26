import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import os

class PatchDataset(Dataset):
    def __init__(self, path, train=True, transform=None):
        self.patches = [path + f for f in os.listdir(path)]
        
        n = int(len(self.patches) * 0.8)
        self.patches = self.patches[:n] if train else self.patches[n:]
        
        self.transform = transform

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        '''
        [In_tissue, Total, G1, G2, ...] 
        shape : 1+1+N, W, H 
        '''

        patch = random.choice(self.patches)
        data = train.load_data(patch)
        import pandas as pd

        data.adata.obs['n_counts'] = data.adata.X.sum(axis=1)
        df = data.adata.obs[['array_row', 'array_col', 'in_tissue', 'n_counts']]
        epcam = data.adata.to_df()['EPCAM']
        df = pd.concat([df, epcam], axis=1)

        df_rasterize = sthdviz.rasterize_numerical(df, 'EPCAM')
        df_in_tissue = sthdviz.rasterize_numerical(df, 'in_tissue')
        df_n_counts = sthdviz.rasterize_numerical(df, 'n_counts')
        
        block = np.stack([df_in_tissue, df_n_counts, df_rasterize], axis=0)
        block = torch.from_numpy(block)
        return self.transform(block), 0

class Pad:
    def __init__(self, crop_size, padding_mode='constant'):
        self.crop_size = crop_size
        self.padding_mode = padding_mode

    def __call__(self, img):
        _, width, height = img.shape
        pad_w = max(0, self.crop_size[0] - width)
        pad_h = max(0, self.crop_size[1] - height)

        if pad_w > 0 or pad_h > 0:
            padding = (pad_h//2, pad_h-pad_h//2, pad_w//2, pad_w-pad_w//2)
            img = F.pad(img, padding, mode=self.padding_mode, value=0)
    
        return img

class Poisson:
    def __init__(self, ratio_min, ratio_max):
        self.ratio = (ratio_min, ratio_max)

    def __call__(self, img):
        ratio = torch.empty(1).uniform_(*self.ratio)
        density = img * ratio
        counts = torch.poisson(density)
        bounds = (img > 0.5).float()
        return torch.cat([bounds, density, counts, counts], dim=0)

class SimulatedDataset(Dataset):
    def __init__(self, size, field, p=0.1, train=True, transform=None):
        from simulate.simulate_count import create_matrix
        self.p = 0.1
        self.transform = transform
        self.filtered = field
        self.n_celltypes, self.n_gene, self.matrix = create_matrix(self.filtered)
        self.size = size

        print(self.n_celltypes, self.n_gene)

    def __len__(self):
        return int(1e8)

    def __getitem__(self, idx):
        from simulate.simulate_count import create_ct, diffuse_adata
        from STHD import sthdviz
        data, ct_mask = create_ct(self.size, self.size, n_celltypes=self.n_celltypes, matrix=self.matrix, ncells=3, cell_r_range=(self.size//4, self.size//2))
        # TODO: Diffuse counts # adata = diffuse_adata(adata, p=self.p)
        
        block = data.transpose(2, 0, 1)
        in_tissue = (ct_mask != self.n_celltypes-1)
        density = total = block.sum(axis=0)
        info = np.stack([in_tissue, density, total], axis=0)

        block = torch.from_numpy(np.vstack([info, block]))
        
        return self.transform(block), ct_mask


def get_dataset(config):

    # Compute batch size for this worker.
    batch_size = config.training.batch_size
    if batch_size % torch.cuda.device_count() != 0:
        raise ValueError(f'Batch sizes {batch_size} must be divided by'
                         f'the number of devices ({torch.cuda.device_count()})')

    # Reduce this when image resolution is too large and data pointer is stored
    shuffle_buffer_size = 10000
    train_dataset = test_dataset = None

    # Create dataset builders for each dataset.
    if config.data.dataset == 'MNIST':
        transform = transforms.Compose([transforms.Resize(config.data.image_size),
                                        transforms.ToTensor(),
                                        Poisson(config.data.poisson_ratio_min, config.data.poisson_ratio_max),
                                        transforms.RandomAffine(degrees=90, scale=(0.8, 1.2), translate=(0.2, 0.2)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.GaussianBlur(kernel_size=5, sigma=(config.data.pre_blur, config.data.pre_blur))])
                                        # TODO: Separate the blurring 

        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    elif config.data.dataset == 'VISIUM':
        transform = transforms.Compose([Pad(config.data.image_size),
                                        transforms.RandomCrop(config.data.image_size),
                                        transforms.RandomAffine(degrees=90, scale=(0.8, 1.2), translate=(0.2, 0.2)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.GaussianBlur(kernel_size=5, sigma=(config.data.pre_blur, config.data.pre_blur))])

        train_dataset = PatchDataset(path, transform=transform)
        test_dataset = PatchDataset(path, transform=transform) # TODO: define test dataset

    elif config.data.dataset == 'SIMULATE':
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.GaussianBlur(kernel_size=5, sigma=(config.data.pre_blur, config.data.pre_blur))])

        train_dataset = SimulatedDataset(config.data.image_size, config.data.field, transform=transform)
        test_dataset = SimulatedDataset(config.data.image_size, config.data.field, transform=transform) # TODO: define test dataset

    else:
        raise NotImplementedError(
            f'Dataset {config.data.dataset} not yet supported.')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

    return train_loader, test_loader


if __name__ == '__main__':
    from config.default_configs import get_default_configs
    config = get_default_configs()

    train_loader, test_loader = get_dataset(config)
    data, _ = next(iter(test_loader))

    for i, batch in enumerate(train_loader):
        data, _ = batch
        break

    import matplotlib.pyplot as plt
    fig, axe = plt.subplots(nrows=1, ncols=4, figsize=(25, 25))
    for i in range(4):
        axe[i].imshow(data[i, 0].cpu(), interpolation='nearest')
    plt.show()
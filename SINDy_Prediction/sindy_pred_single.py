import argparse
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import pysindy as ps
import random
import os
import json
import matplotlib.pyplot as plt
import matplotlib as mpl

newcmp = mpl.colormaps['gist_earth_r']

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def l1_penalty(params, lamb):
    return lamb * sum(p.abs().sum() for p in params)

class CNNModel(nn.Module):
    def __init__(self, input_channels=30, output_channels=1, dt=0.1, l1_alpha=0.01):
        super(CNNModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, bias=False)
        self.mse_loss = nn.MSELoss()
        self.l1_alpha = l1_alpha
        self.dt = dt

    def forward(self, feature, z_dot=None, z_t=None, z_t_add_1=None, inference=False):
        z_dot_pre = self.conv(feature)
        if not inference:
            loss = self.mse_loss(z_dot, z_dot_pre) \
                 + self.mse_loss(z_t + z_dot_pre * self.dt, z_t_add_1) \
                 + l1_penalty(self.parameters(), self.l1_alpha)
            return z_dot_pre, loss
        else:
            return z_dot_pre

def train(model, train_dataloader, test_dataloader, optimizer, epoches=200):
    model.train()
    best_model_wts = model.state_dict()
    min_loss = torch.inf
    train_loss_list = []
    test_loss_list = []
    for epoch in range(epoches):
        losses = 0.0
        for feature, z_dot, z_t, z_t_add_1 in train_dataloader:
            feature, z_dot, z_t, z_t_add_1 = feature.to(device), z_dot.to(device), z_t.to(device), z_t_add_1.to(device)
            _, loss = model(feature, z_dot, z_t, z_t_add_1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

        test_loss = test(model, test_dataloader)
        train_loss_list.append(losses)
        test_loss_list.append(test_loss)
        print(f"[Epoch {epoch+1:4d}] Train Loss: {losses:.6f}, Test Loss: {test_loss:.6f}, Min Test Loss: {min_loss:.6f}")
        if test_loss < min_loss:
            best_model_wts = model.state_dict()
            min_loss = test_loss

    return best_model_wts, train_loss_list, test_loss_list

def test(model, test_dataloader):
    model.eval()
    with torch.no_grad():
        losses = 0.0
        for feature, z_dot, z_t, z_t_add_1 in test_dataloader:
            feature, z_dot, z_t, z_t_add_1 = feature.to(device), z_dot.to(device), z_t.to(device), z_t_add_1.to(device)
            _, loss = model(feature, z_dot, z_t, z_t_add_1)
            losses += loss.item()
    return losses / len(test_dataloader)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy_path', type=str, required=True, help='Path to input .npy file')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--steps_predict', type=int, default=23)
    parser.add_argument('--epoches', type=int, default=1000)
    parser.add_argument('--l1_alpha', type=float, default=1e-2)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    setup_seed(20)
    device = args.device
    mkdir(args.save_dir)

    npy_path = args.npy_path
    save_dir = args.save_dir
    steps_predict = args.steps_predict
    epoches = args.epoches
    l1_alpha = args.l1_alpha
    lr = args.lr
    batch_size = args.batch_size

    library_functions = [lambda z: z, lambda z: z*z, lambda z: np.sin(z), lambda z: np.cos(z)]
    library_function_names = [lambda z: z, lambda z: z+z, lambda z: f'sin({z})', lambda z: f'cos({z})']

    data_grid = np.load(npy_path)
    z = np.transpose(data_grid, (2, 1, 0))
    x = np.arange(0, z.shape[0]) * 0.1
    y = np.arange(0, z.shape[1]) * 0.1
    t = np.arange(0, z.shape[2]) * 0.1
    dx, dy, dt = x[1]-x[0], y[1]-y[0], t[1]-t[0]

    z = np.expand_dims(z, axis=3)
    z_dot = ps.SmoothedFiniteDifference(axis=2)._differentiate(z, dt)
    X, Y = np.meshgrid(x, y)
    spatial_grid = np.asarray([X, Y]).T

    pde_lib = ps.PDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=2,
        spatial_grid=spatial_grid,
        include_bias=True,
        is_uniform=True,
        periodic=True
    )
    pde_lib.fit(z)
    feature = pde_lib.transform(z)

    features = torch.tensor(np.transpose(feature, (2, 3, 0, 1)), dtype=torch.float)
    z_dot = torch.tensor(np.transpose(z_dot, (2, 3, 0, 1)), dtype=torch.float)
    z = torch.tensor(np.transpose(z, (2, 3, 0, 1)), dtype=torch.float)
    z_input = z

    train_dataset = TensorDataset(features[:steps_predict-1], z_dot[:steps_predict-1], z[:steps_predict-1], z[1:steps_predict])
    test_dataset = TensorDataset(features[steps_predict-1:-1], z_dot[steps_predict-1:-1], z[steps_predict-1:-1], z[steps_predict:])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    success = False
    while not success:
        model = CNNModel(features.shape[1], z_dot.shape[1], dt=dt, l1_alpha=l1_alpha).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        best_model_wts, train_loss_list, test_loss_list = train(model, train_dataloader, test_dataloader, optimizer, epoches)

        print('\nPredicting...')
        model.load_state_dict(best_model_wts)
        model.eval()
        with torch.no_grad():
            start = torch.tensor(np.expand_dims(z_input[-steps_predict-1], 0)).to(device)
            print(start.min(), start.max())
            z_pred_list = [start]
            for i in tqdm(range(steps_predict), 'Predict'):
                z_t = z_pred_list[-1].to(device)
                feature = pde_lib.transform(z_t.permute(2, 3, 0, 1).cpu().numpy())
                feature = torch.tensor(feature, dtype=torch.float).to(device).permute(2, 3, 0, 1)
                z_dot_pred = model(feature, inference=True)
                z_t_add_1 = z_t + z_dot_pred * dt
                if z_t_add_1.max() > 100 or z_t_add_1.min() < -100 or torch.isnan(z_t_add_1).any():
                    break
                z_pred_list.append(z_t_add_1.cpu())

            z_pred_list = z_pred_list[1:]
            print(f'pred_max_steps : {len(z_pred_list)}')

        if len(z_pred_list) < steps_predict:
            print('Predicting failed. Increasing l1_alpha and retraining...')
            l1_alpha += 0.01
        else:
            success = True

    pred_save_dir = os.path.join(save_dir, 'predicted_npy')
    mkdir(pred_save_dir)
    for i in range(len(z_pred_list)):
        itrue = z_input[-steps_predict+i, 0].numpy()
        ipred = z_pred_list[i].squeeze().numpy()
        np.save(os.path.join(pred_save_dir, f'{i+1:02d}_true.npy'), itrue)
        np.save(os.path.join(pred_save_dir, f'{i+1:02d}_pred.npy'), ipred)

    ssims, psnrs, mses = [], [], []
    for i in range(len(z_pred_list)):
        itrue = z_input[-steps_predict+i, 0].numpy()
        ipred = z_pred_list[i].squeeze().numpy()
        itrue = (itrue - data_grid.min()) / (data_grid.max() - data_grid.min())
        ipred = (ipred - data_grid.min()) / (data_grid.max() - data_grid.min())
        ssims.append(ssim(itrue, ipred, data_range=1.))
        psnrs.append(psnr(itrue, ipred, data_range=1.))
        mses.append(mse(itrue, ipred))

    print(f'Mean SSIM: {np.mean(ssims):.4f}, PSNR: {np.mean(psnrs):.2f}, MSE: {np.mean(mses):.6f}')

    idx_list = [0, 5, 10, 15, 19]
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(idx_list):
        if idx >= len(z_pred_list):
            continue
        true_img = z_input[-steps_predict+idx, 0].numpy()
        pred_img = z_pred_list[idx].squeeze().numpy()
        norm = mpl.colors.Normalize(vmin=min(true_img.min(), pred_img.min()), vmax=max(true_img.max(), pred_img.max()))

        plt.subplot(2, len(idx_list), i+1)
        plt.imshow(true_img, cmap=newcmp, norm=norm)
        plt.title(f'True t={idx}')
        plt.axis('off')
        plt.subplot(2, len(idx_list), i+1+len(idx_list))
        plt.imshow(pred_img, cmap=newcmp, norm=norm)
        plt.title(f'Pred t={idx}')
        plt.axis('off')

    plt.savefig(f'{save_dir}/prediction_vis.png')

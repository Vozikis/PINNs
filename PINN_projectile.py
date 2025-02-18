import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.signal import medfilt


class PINN(nn.Module):
    def __init__(self, hidden_dim=20):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2) 
        )

    def forward(self, t):
        return self.net(t)

def physics_loss(model, t_interior, g=9.8):
    t_interior.requires_grad_(True)
    pred = model(t_interior)
    x_pred = pred[:, 0:1]
    y_pred = pred[:, 1:2]

    dx_dt = torch.autograd.grad(
        x_pred, t_interior,
        grad_outputs=torch.ones_like(x_pred),
        create_graph=True
    )[0]
    dy_dt = torch.autograd.grad(
        y_pred, t_interior,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True
    )[0]

    d2x_dt2 = torch.autograd.grad(
        dx_dt, t_interior,
        grad_outputs=torch.ones_like(dx_dt),
        create_graph=True
    )[0]
    d2y_dt2 = torch.autograd.grad(
        dy_dt, t_interior,
        grad_outputs=torch.ones_like(dy_dt),
        create_graph=True
    )[0]

    res_x = d2x_dt2  # x''(t) = 0
    res_y = d2y_dt2 + g  # y''(t) + g = 0
    return torch.mean(res_x**2 + res_y**2)

def data_loss(model, t_data, x_data, y_data):
    pred = model(t_data)
    x_pred = pred[:, 0:1]
    y_pred = pred[:, 1:2]
    return torch.mean((x_pred - x_data)**2 + (y_pred - y_data)**2)


def apply_filter(t_data, x_data, y_data, kernel_size=25):
    """
    Apply a median filter to smooth the x_data and y_data.
    """
    x_data_np = x_data.numpy().flatten()
    y_data_np = y_data.numpy().flatten()

    x_data_filtered_np = medfilt(x_data_np, kernel_size=kernel_size)
    y_data_filtered_np = medfilt(y_data_np, kernel_size=kernel_size)

    x_data_filtered = torch.tensor(x_data_filtered_np, dtype=torch.float32).reshape(-1, 1)
    y_data_filtered = torch.tensor(y_data_filtered_np, dtype=torch.float32).reshape(-1, 1)

    return t_data, x_data_filtered, y_data_filtered

def train_pinn_projectile(T=2.0, g=9.8, n_phys_points=50, t_data=None, x_data=None, y_data=None, n_epochs=5000, lr=1e-3):
    t_interior = torch.linspace(0, T, n_phys_points).reshape(-1, 1)
    model = PINN(hidden_dim=20)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        p_loss = physics_loss(model, t_interior, g=g)
        d_loss = data_loss(model, t_data, x_data, y_data)
        loss_value = p_loss + d_loss
        loss_value.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch} | PDE Loss = {p_loss.item():.6f} | "
                  f"Data Loss = {d_loss.item():.6f} | Total = {loss_value.item():.6f}")

    return model

def load_trajectories(base_directory, file_name, kernel_size=25, apply_smoothing=False):
    all_trajectories = []
    for trajectory_number in range(50):
        video_dir = os.path.join(base_directory, f"video_{trajectory_number}")
        file_path = os.path.join(video_dir, file_name)
        
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            
            data_len = len(data)
            t = np.arange(data_len)
            x = np.array([item[0] if item is not None else np.nan for item in data])
            y = np.array([item[1] if item is not None else np.nan for item in data])
            
            valid_start_idx = np.where(~np.isnan(x))[0][0]
            x = x[valid_start_idx:]
            y = y[valid_start_idx:]
            t = t[valid_start_idx:]
            
            df = pd.DataFrame({'t': t, 'x': x, 'y': y}).interpolate(method='linear')
            traj_x = -df['x'].to_numpy() / 100.0
            traj_y = -df['y'].to_numpy() / 100.0
            traj_t = df['t'].to_numpy()
            
            # if trajectory_number ==0:
            #     traj_x = traj_x[20:80]
            #     traj_y = traj_y[20:80]
            # if trajectory_number ==1:
            #     traj_x = traj_x[20:80]
            #     traj_y = traj_y[20:80]
            # if trajectory_number ==2:
            #     traj_x = traj_x[20:80]
            #     traj_y = traj_y[20:80]
            # if trajectory_number ==3:
            #     traj_x = traj_x[20:80]
            #     traj_y = traj_y[20:80]
            # if trajectory_number ==4:
            #     traj_x = traj_x[20:80]
            #     traj_y = traj_y[20:80]
            # if trajectory_number ==5:
            #     traj_x = traj_x[20:80]
            #     traj_y = traj_y[20:80]
            # if trajectory_number ==6:
            #     traj_x = traj_x[20:80]
            #     traj_y = traj_y[20:80]
            # if trajectory_number ==7:
            #     traj_x = traj_x[20:80]
            #     traj_y = traj_y[20:80]
            # if trajectory_number ==8:
            #     traj_x = traj_x[20:80]
            #     traj_y = traj_y[20:80]
            # if trajectory_number ==9:
            #     traj_x = traj_x[20:80]
            #     traj_y = traj_y[20:80]
            # if trajectory_number ==10:
            #     traj_x = traj_x[20:80]
            #     traj_y = traj_y[20:80]
            # if trajectory_number ==11:
            #     traj_x = traj_x[20:80]
            #     traj_y = traj_y[20:80]
            # if trajectory_number ==12:
            #     traj_x = traj_x[20:80]
            #     traj_y = traj_y[20:80]
            # if trajectory_number ==13:
            #     traj_x = traj_x[20:80]
            #     traj_y = traj_y[20:80]
            # if trajectory_number ==14:
            #     traj_x = traj_x[20:80]
            #     traj_y = traj_y[20:80]
            # if trajectory_number ==15:
            #     traj_x = traj_x[20:80]
            #     traj_y = traj_y[20:80]
            # if trajectory_number ==16:
            #     traj_x = traj_x[20:80]
            #     traj_y = traj_y[20:80]
            # if trajectory_number ==17:
            #     traj_x = traj_x[20:80]
            #     traj_y = traj_y[20:80]
            # if trajectory_number ==18:
            #     traj_x = traj_x[20:80]
            #     traj_y = traj_y[20:80]
            # if trajectory_number ==19:
            #     traj_x = traj_x[20:80]
            #     traj_y = traj_y[20:80]
            # if trajectory_number ==20:
            #     traj_x = traj_x[20:80]
            #     traj_y = traj_y[20:80]
            # if trajectory_number ==21:
            #     traj_x = traj_x[20:80]
            #     traj_y = traj_y[20:80]


            traj_t_shifted = np.linspace(0, 6, len(traj_x)) 
            t_data = torch.tensor(traj_t_shifted, dtype=torch.float32).reshape(-1, 1)
            x_data = torch.tensor(traj_x, dtype=torch.float32).reshape(-1, 1)
            y_data = torch.tensor(traj_y, dtype=torch.float32).reshape(-1, 1)

            if apply_smoothing:
                t_data, x_data, y_data = apply_filter(t_data, x_data, y_data, kernel_size=kernel_size)

            all_trajectories.append((t_data, x_data, y_data))
    return all_trajectories

def train_and_evaluate_trajectories(all_trajectories):
    mse_list = []
    nmse_list = []
    mape_list = []

    for i, (t_data, x_data, y_data) in enumerate(all_trajectories):
        print(f"Training PINN for Trajectory {i + 1}...")
        model = train_pinn_projectile(
            T=2.0,
            g=9.8,
            n_phys_points=50,
            t_data=t_data,
            x_data=x_data,
            y_data=y_data,
            n_epochs=100000,
            lr=1e-3
        )

        t_test = torch.linspace(0, 6, 200).reshape(-1, 1)
        pred = model(t_test).detach().numpy()
        x_pred = pred[:, 0]
        y_pred = pred[:, 1]

        x_true = np.interp(t_test.numpy().flatten(), t_data.numpy().flatten(), x_data.numpy().flatten())
        y_true = np.interp(t_test.numpy().flatten(), t_data.numpy().flatten(), y_data.numpy().flatten())

        err_x = (x_pred - x_true)
        err_y = (y_pred - y_true)
        mse_xy = np.mean(err_x**2 + err_y**2)
        var_xy = np.var(np.concatenate([x_true, y_true]))
        nmse_xy = mse_xy / var_xy if var_xy > 1e-12 else 0.0

        denom_x = np.where(np.abs(x_true) < 1e-6, 1.0, x_true)
        denom_y = np.where(np.abs(y_true) < 1e-6, 1.0, y_true)
        mape_x = np.mean(np.abs(err_x / denom_x)) * 100
        mape_y = np.mean(np.abs(err_y / denom_y)) * 100
        mape_xy = 0.5 * (mape_x + mape_y)

        mse_list.append(mse_xy)
        nmse_list.append(nmse_xy)
        mape_list.append(mape_xy)

        print(f"\nTrajectory {i + 1} Metrics:")
        print(f"  MSE (x,y combined): {mse_xy:.6f}")
        print(f"  NMSE (x,y combined): {nmse_xy:.6f}")
        print(f"  MAPE (x,y avg): {mape_xy:.2f}%")

        plt.figure(figsize=(8, 5))
        plt.subplot(2, 1, 1)
        plt.plot(t_test, x_true, 'b-', label='True X', linewidth=2)
        plt.plot(t_test, x_pred, 'r--', label='Pred X')
        plt.scatter(t_data, x_data, color='k', s=20, label='Data X')
        plt.legend()
        plt.title(f"Trajectory {i + 1} - X(t) | MSE={mse_xy:.6f} | NMSE={nmse_xy:.6f}")

        plt.subplot(2, 1, 2)
        plt.plot(t_test, y_true, 'b-', label='True Y', linewidth=2)
        plt.plot(t_test, y_pred, 'r--', label='Pred Y')
        plt.scatter(t_data, y_data, color='k', s=20, label='Data Y')
        plt.legend()
        plt.title(f"Trajectory {i + 1} - Y(t) | MSE={mse_xy:.6f} | NMSE={nmse_xy:.6f}")

        plt.tight_layout()
        output_dir = "cogvideo_x_projectile_regular_conditioned_9_frames"
        os.makedirs(output_dir, exist_ok=True)
        plot_filename = os.path.join(output_dir, f"cogvideo_x_projectile_regular_conditioned_9_frames_{i+1}.png")
        plt.savefig(plot_filename)
        plt.close()

    avg_mse = np.mean(mse_list)
    avg_nmse = np.mean(nmse_list)
    avg_mape = np.mean(mape_list)

    print("\nAverage Metrics Across All Trajectories:")
    print(f"  Average MSE: {avg_mse:.6f}")
    print(f"  Average NMSE: {avg_nmse:.6f}")
    print(f"  Average MAPE: {avg_mape:.2f}%")

    return avg_mse, avg_nmse, avg_mape

if __name__ == "__main__":
    base_directory = "VGM_trajectories/filtered_trajectories/multi_frame_generations/cogvideo_x/projectile_regular_conditioned_9_frames"
    file_name = "3D_centers.pkl"

    all_trajectories = load_trajectories(base_directory, file_name, apply_smoothing=True, kernel_size=11)
    average_mse, average_nmse, average_mape = train_and_evaluate_trajectories(all_trajectories)

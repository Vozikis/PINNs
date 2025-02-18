import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.signal import medfilt

def apply_filter(t_data, x_data, kernel_size=5):
    
    x_data_np = x_data.numpy().flatten()
    x_data_filtered_np = medfilt(x_data_np, kernel_size=kernel_size)
    x_data_filtered = torch.tensor(x_data_filtered_np, dtype=torch.float32).reshape(-1, 1)
    
    return t_data, x_data_filtered


class PINN(nn.Module):
    def __init__(self, hidden_dim=20):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, t):
        return self.net(t)


def physics_loss(model, t_interior, g=9.8, length=1.0):

    t_interior.requires_grad_(True)
    theta_pred = model(t_interior)

    dtheta_dt = torch.autograd.grad(
        theta_pred,
        t_interior,
        grad_outputs=torch.ones_like(theta_pred),
        create_graph=True
    )[0]

    d2theta_dt2 = torch.autograd.grad(
        dtheta_dt,
        t_interior,
        grad_outputs=torch.ones_like(dtheta_dt),
        create_graph=True
    )[0]

    # Nonlinear pendulum ODE: θ'' + (g/length) * sin(θ) = 0
    omega_sq = g / length
    physics_residual = d2theta_dt2 + omega_sq * torch.sin(theta_pred)

    return torch.mean(physics_residual**2)


def data_loss(model, t_data, y_data):
    y_pred = model(t_data)
    return torch.mean((y_pred - y_data)**2)


def train_pinn_free_fall_discover_IC(
    T=2.0,
    g=9.8,
    length=1.0,
    n_phys_points=50,
    t_data=None,
    y_data=None,
    n_epochs=5000,
    lr=1e-3
):
    t_interior = torch.linspace(0, T, n_phys_points).reshape(-1, 1)
    model = PINN(hidden_dim=20)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        p_loss = physics_loss(model, t_interior, g=g, length=length)
        d_loss = data_loss(model, t_data, y_data)
        loss_value = p_loss + d_loss
        loss_value.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch} | PDE Loss = {p_loss.item():.6f} | "
                  f"Data Loss = {d_loss.item():.6f} | Total = {loss_value.item():.6f}")

    return model

def load_trajectories(base_directory, file_name):
    all_trajectories = []
    for trajectory_number in range(15):
        video_dir = os.path.join(base_directory, f"video_{trajectory_number}")
        file_path = os.path.join(video_dir, file_name)
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}. Skipping trajectory {trajectory_number}.")
            continue
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}. Skipping trajectory {trajectory_number}.")
            continue
        
        data_len = len(data)
        t = np.arange(data_len)
        x = np.array([item[0] if item is not None else np.nan for item in data])
        y = np.array([item[1] if item is not None else np.nan for item in data])
        
        valid_start_idx_x = np.where(~np.isnan(x))[0][0]
        x = x[valid_start_idx_x:]
        y = y[valid_start_idx_x:]
        t = t[valid_start_idx_x:]
        
        df = pd.DataFrame({'t': t, 'x': x, 'y': y}).interpolate(method='linear')
        traj_x = df['y'][:].to_numpy() / 100
        traj_t = df['t'][:].to_numpy()
        
        
        # if trajectory_number == 0:
        #     traj_x = traj_x[80:]
        # elif trajectory_number == 1:
        #     traj_x = traj_x[150:]
        # elif trajectory_number == 2:
        #     traj_x = traj_x[120:] 
        # elif trajectory_number == 3:
        #     traj_x = traj_x[80:]
        # elif trajectory_number == 4:
        #     traj_x = traj_x[80:] 
        # elif trajectory_number == 5:
        #     traj_x = traj_x[80:]
        # elif trajectory_number == 6:
        #     traj_x = traj_x[50:] 
        # elif trajectory_number == 7:
        #     traj_x = traj_x[50:]
        # elif trajectory_number == 8:
        #     traj_x = traj_x[50:]
        # elif trajectory_number == 10:
        #     traj_x = traj_x[60:]
        # elif trajectory_number == 11:
        #     traj_x = traj_x[80:]
        # elif trajectory_number == 12:
        #     traj_x = traj_x[80:] 
        # elif trajectory_number == 13:
        #     traj_x = traj_x[50:]
        # elif trajectory_number == 14:
        #     traj_x = traj_x[40:]
        
        traj_t_shifted = np.linspace(0, 19, len(traj_x)) 
        t_data = torch.tensor(traj_t_shifted, dtype=torch.float32).reshape(-1, 1)
        x_data = torch.tensor(traj_x, dtype=torch.float32).reshape(-1, 1)

        all_trajectories.append((t_data, x_data))
    filtered_trajectories = []
    for t_data, x_data in all_trajectories:
        t_data_filtered, x_data_filtered = apply_filter(t_data, x_data, kernel_size=25)
        filtered_trajectories.append((t_data_filtered, x_data_filtered))

    return filtered_trajectories


def train_and_evaluate_trajectories(all_trajectories):
    mse_list = []
    nmse_list = []

    for i, (t_data, x_data) in enumerate(all_trajectories):
        print(f"Training PINN for Trajectory {i + 1}...")
        model = train_pinn_free_fall_discover_IC(
            T=2.0,
            g=9.8,
            length=3,
            n_phys_points=200,
            t_data=t_data,
            y_data=x_data,
            n_epochs=200000,  
            lr=1e-3
        )

        t_test = torch.linspace(0, 19, 200).reshape(-1, 1)
        y_pred = model(t_test).detach().numpy().flatten()
        y_true = np.interp(t_test.numpy().flatten(), t_data.numpy().flatten(), x_data.numpy().flatten())

        mse = np.mean((y_pred - y_true) ** 2)
        variance_y_true = np.var(y_true)
        nmse = mse / variance_y_true

        mse_list.append(mse)
        nmse_list.append(nmse)

        print(f"\nTrajectory {i + 1} Metrics:")
        print(f"  Mean Squared Error (MSE): {mse:.6f}")
        print(f"  Normalized MSE (NMSE): {nmse:.6f}")

        plt.figure(figsize=(7, 5))
        plt.plot(t_test.numpy().flatten(), y_true, label="True Trajectory", color='blue', linewidth=2)
        plt.plot(t_test.numpy().flatten(), y_pred, label="Predicted Trajectory", linestyle='--', color='red')
        plt.scatter(t_data.numpy(), x_data.numpy(), label="Data Points", color='black', s=40)

        plt.title(
            f"Trajectory {i + 1} - Predicted vs Real\n"
            f"MSE: {mse:.6f}, NMSE: {nmse:.6f}",
            fontsize=10
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")
        plt.legend()
        plt.grid()
        plt.tight_layout()

        output_dir = "Non_holonomic_pendulum"
        os.makedirs(output_dir, exist_ok=True)

        plot_filename = os.path.join(output_dir, f"Non_holonomic_pendulum_{i + 1}.png")
        plt.savefig(plot_filename)
        plt.close()

    avg_mse = np.mean(mse_list)
    avg_nmse = np.mean(nmse_list)

    print(f"\nAverage Metrics Across All Trajectories:")
    print(f"  Average MSE: {avg_mse:.6f}")
    print(f"  Average NMSE: {avg_nmse:.6f}")

    return avg_mse, avg_nmse


if __name__ == "__main__":
    base_directory = "tracking-centers/tracking_centers/non-holonomic_pendulum/30_10_2024/fps_30"
    file_name = "3D_centers.pkl"

    all_trajectories = all_trajectories = [(t_data, x_data - torch.mean(x_data)) for t_data, x_data in load_trajectories(base_directory, file_name)]
    average_mse, average_nmse = train_and_evaluate_trajectories(all_trajectories)

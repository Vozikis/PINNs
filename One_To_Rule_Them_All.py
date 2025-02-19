import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.signal import medfilt

def preprocess_directories(base_directory):
    """
    Rename all subdirectories in the base_directory sequentially as video_0, video_1, etc.
    """
    subdirs = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    subdirs_sorted = sorted(subdirs)
    for i, subdir in enumerate(subdirs_sorted):
        old_path = os.path.join(base_directory, subdir)
        new_name = f"video_{i}"
        new_path = os.path.join(base_directory, new_name)
        if old_path != new_path:
            try:
                os.rename(old_path, new_path)
                print(f"Renamed '{old_path}' to '{new_path}'")
            except Exception as e:
                print(f"Error renaming {old_path} to {new_path}: {e}")

def apply_filter(t_data, x_data, y_data=None, kernel_size=25):
    """
    Apply median filtering. If y_data is provided, both x and y are filtered.
    Otherwise only x_data is filtered.
    """
    if y_data is None:
        x_np = x_data.numpy().flatten()
        x_filtered_np = medfilt(x_np, kernel_size=kernel_size)
        x_filtered = torch.tensor(x_filtered_np, dtype=torch.float32).reshape(-1, 1)
        return t_data, x_filtered
    else:
        x_np = x_data.numpy().flatten()
        y_np = y_data.numpy().flatten()
        x_filtered_np = medfilt(x_np, kernel_size=kernel_size)
        y_filtered_np = medfilt(y_np, kernel_size=kernel_size)
        x_filtered = torch.tensor(x_filtered_np, dtype=torch.float32).reshape(-1, 1)
        y_filtered = torch.tensor(y_filtered_np, dtype=torch.float32).reshape(-1, 1)
        return t_data, x_filtered, y_filtered

# PENDULUM PINN
class PendulumPINN(nn.Module):
    def __init__(self, hidden_dim=20):
        super(PendulumPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, t):
        return self.net(t)

    def physics_loss(self, t_interior, g=9.8, length=1.0):
        t_interior.requires_grad_(True)
        theta_pred = self.forward(t_interior)
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
        physics_residual = d2theta_dt2 + (g/length) * torch.sin(theta_pred)
        return torch.mean(physics_residual**2)

    def data_loss(self, t_data, y_data):
        y_pred = self.forward(t_data)
        return torch.mean((y_pred - y_data)**2)

    @staticmethod
    def train_model(t_data, y_data, T=2.0, g=9.8, length=1.0, n_phys_points=200, 
                    n_epochs=5000, lr=1e-3, early_stop_patience=500, early_stop_min_delta=1e-6):
        t_interior = torch.linspace(0, T, n_phys_points).reshape(-1, 1)
        model = PendulumPINN(hidden_dim=20)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            p_loss = model.physics_loss(t_interior, g=g, length=length)
            d_loss = model.data_loss(t_data, y_data)
            loss_value = p_loss + d_loss
            loss_value.backward()
            optimizer.step()
            
            current_loss = loss_value.item()
            if current_loss < best_loss - early_stop_min_delta:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 500 == 0:
                print(f"[Pendulum] Epoch {epoch:06d} | PDE Loss = {p_loss.item():.6f} | Data Loss = {d_loss.item():.6f} | Total = {current_loss:.6f}")

            if patience_counter >= early_stop_patience:
                print(f"[Pendulum] Early stopping triggered at epoch {epoch} with best loss {best_loss:.6f}.")
                break
                
        return model

    @staticmethod
    def load_trajectories(base_directory, file_name):
        trajectories = []
        for traj_num in range(200):
            video_dir = os.path.join(base_directory, f"video_{traj_num}")
            file_path = os.path.join(video_dir, file_name)
            if not os.path.exists(file_path):
                print(f"[Pendulum] File not found: {file_path}. Skipping trajectory {traj_num}.")
                continue
            try:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
            except Exception as e:
                print(f"[Pendulum] Error loading {file_path}: {e}. Skipping trajectory {traj_num}.")
                continue

            data_len = len(data)
            t = np.arange(data_len)
            x = np.array([item[0] if item is not None else np.nan for item in data])
            y = np.array([item[1] if item is not None else np.nan for item in data])
            valid_idx = np.where(~np.isnan(x))[0]
            if len(valid_idx) == 0:
                continue
            start = valid_idx[0]
            x = x[start:]
            y = y[start:]
            t = t[start:]
            df = pd.DataFrame({'t': t, 'x': x, 'y': y}).interpolate(method='linear')
            traj = df['y'].to_numpy() / 100.0 
            traj_t = np.linspace(0, 19, len(traj))
            t_data = torch.tensor(traj_t, dtype=torch.float32).reshape(-1, 1)
            x_data = torch.tensor(traj, dtype=torch.float32).reshape(-1, 1)
            trajectories.append((t_data, x_data))
        return trajectories

    @staticmethod
    def evaluate(trajectories, output_dir):
        results = []
        mse_list, nmse_list = [], []
        os.makedirs(output_dir, exist_ok=True)

        for i, (t_data, x_data) in enumerate(trajectories):
            print(f"\n[Pendulum] Training PINN for Trajectory {i+1}...")
            model = PendulumPINN.train_model(
                t_data=t_data, y_data=x_data, T=2.0, g=9.8, length=1.0, 
                n_phys_points=200, n_epochs=200000, lr=1e-3
            )
            t_test = torch.linspace(0, 19, 200).reshape(-1, 1)
            y_pred = model(t_test).detach().numpy().flatten()
            y_true = np.interp(t_test.numpy().flatten(), t_data.numpy().flatten(), x_data.numpy().flatten())
            mse = np.mean((y_pred - y_true) ** 2)
            nmse = mse / np.var(y_true)
            mse_list.append(mse)
            nmse_list.append(nmse)
            results.append({"Trajectory": i+1, "MSE": mse, "NMSE": nmse})
            print(f"Trajectory {i+1} Metrics: MSE: {mse:.6f} | NMSE: {nmse:.6f}")
            plt.figure(figsize=(7, 5))
            plt.plot(t_test.numpy().flatten(), y_true, label="True Trajectory", color='blue', linewidth=2)
            plt.plot(t_test.numpy().flatten(), y_pred, label="Predicted Trajectory", linestyle='--', color='red')
            plt.scatter(t_data.numpy(), x_data.numpy(), label="Data Points", color='black', s=40)
            plt.xlabel("Time (s)")
            plt.ylabel("Position (m)")
            plt.title(f"Pendulum Trajectory\nMSE: {mse:.6f} | NMSE: {nmse:.6f}")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"pendulum_traj_{i+1}.png"))
            plt.close()

        avg_mse = np.mean(mse_list)
        avg_nmse = np.mean(nmse_list)
        results.append({"Trajectory": "Average", "MSE": avg_mse, "NMSE": avg_nmse})
        results_df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "pendulum_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\n[Pendulum] Average MSE: {avg_mse:.6f} | Average NMSE: {avg_nmse:.6f}")
        print(f"Results saved to {csv_path}")
        return avg_mse, avg_nmse

# FREE FALL PINN
class FreeFallPINN(nn.Module):
    def __init__(self, hidden_dim=20):
        super(FreeFallPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, t):
        return self.net(t)

    def physics_loss(self, t_interior, g=9.8):
        t_interior.requires_grad_(True)
        y_pred = self.forward(t_interior)
        dydt = torch.autograd.grad(
            y_pred,
            t_interior,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True
        )[0]
        d2ydt2 = torch.autograd.grad(
            dydt,
            t_interior,
            grad_outputs=torch.ones_like(dydt),
            create_graph=True
        )[0]
        return torch.mean((d2ydt2 + g)**2)

    def data_loss(self, t_data, y_data):
        y_pred = self.forward(t_data)
        return torch.mean((y_pred - y_data)**2)

    @staticmethod
    def train_model(t_data, y_data, T=2.0, g=9.8, n_phys_points=50, 
                    n_epochs=5000, lr=1e-3, early_stop_patience=500, early_stop_min_delta=1e-6):
        t_interior = torch.linspace(0, T, n_phys_points).reshape(-1, 1)
        model = FreeFallPINN(hidden_dim=20)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            p_loss = model.physics_loss(t_interior, g=g)
            d_loss = model.data_loss(t_data, y_data)
            loss_value = p_loss + d_loss
            loss_value.backward()
            optimizer.step()
            
            current_loss = loss_value.item()
            if current_loss < best_loss - early_stop_min_delta:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 500 == 0:
                print(f"[Free Fall] Epoch {epoch:06d} | PDE Loss = {p_loss.item():.6f} | Data Loss = {d_loss.item():.6f} | Total = {current_loss:.6f}")

            if patience_counter >= early_stop_patience:
                print(f"[Free Fall] Early stopping triggered at epoch {epoch} with best loss {best_loss:.6f}.")
                break
                
        return model

    @staticmethod
    def load_trajectories(base_directory, file_name):
        trajectories = []
        for traj_num in range(200):
            video_dir = os.path.join(base_directory, f"video_{traj_num}")
            file_path = os.path.join(video_dir, file_name)
            if not os.path.exists(file_path):
                continue
            try:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
            except Exception as e:
                print(f"[Free Fall] Error loading trajectory {traj_num}: {e}")
                continue
            data_len = len(data)
            t = np.arange(data_len)
            x = np.array([item[0] if item is not None else np.nan for item in data])
            y = np.array([item[1] if item is not None else np.nan for item in data])
            valid_idx = np.where(~np.isnan(x))[0]
            if len(valid_idx) == 0:
                continue
            start = valid_idx[0]
            x = x[start:]
            y = y[start:]
            t = t[start:]
            df = pd.DataFrame({'t': t, 'x': x, 'y': y}).interpolate(method='linear')
            traj = -df['x'].to_numpy() / 100.0
            traj_t = np.linspace(0, 1, len(traj))
            t_data = torch.tensor(traj_t, dtype=torch.float32).reshape(-1, 1)
            x_data = torch.tensor(traj, dtype=torch.float32).reshape(-1, 1)
            trajectories.append((t_data, x_data))
        return trajectories

    @staticmethod
    def evaluate(trajectories, output_dir):
        results = []
        mse_list, nmse_list = [], []
        os.makedirs(output_dir, exist_ok=True)

        for i, (t_data, x_data) in enumerate(trajectories):
            print(f"\n[Free Fall] Training PINN for Trajectory {i+1}...")
            model = FreeFallPINN.train_model(
                t_data=t_data, y_data=x_data, T=2.0, g=9.8, n_phys_points=50,
                n_epochs=200000, lr=1e-3
            )
            t_test = torch.linspace(0, 1, 200).reshape(-1, 1)
            y_pred = model(t_test).detach().numpy().flatten()
            y_true = np.interp(t_test.numpy().flatten(), t_data.numpy().flatten(), x_data.numpy().flatten())
            mse = np.mean((y_pred - y_true)**2)
            nmse = mse / np.var(y_true)
            mse_list.append(mse)
            nmse_list.append(nmse)
            results.append({"Trajectory": i+1, "MSE": mse, "NMSE": nmse})
            print(f"Trajectory {i+1} Metrics: MSE: {mse:.6f} | NMSE: {nmse:.6f}")
            plt.figure(figsize=(7, 5))
            plt.plot(t_test.numpy().flatten(), y_true, label="True Trajectory", color='blue', linewidth=2)
            plt.plot(t_test.numpy().flatten(), y_pred, label="Predicted Trajectory", linestyle='--', color='red')
            plt.scatter(t_data.numpy(), x_data.numpy(), label="Data Points", color='black', s=40)
            plt.xlabel("Time (s)")
            plt.ylabel("Position (m)")
            plt.title(f"Free Fall Trajectory\nMSE: {mse:.6f} | NMSE: {nmse:.6f}")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"freefall_traj_{i+1}.png"))
            plt.close()

        avg_mse = np.mean(mse_list)
        avg_nmse = np.mean(nmse_list)
        results.append({"Trajectory": "Average", "MSE": avg_mse, "NMSE": avg_nmse})
        results_df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "freefall_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\n[Free Fall] Average MSE: {avg_mse:.6f} | Average NMSE: {avg_nmse:.6f}")
        print(f"Results saved to {csv_path}")
        return avg_mse, avg_nmse

# PROJECTILE PINN
class ProjectilePINN(nn.Module):
    def __init__(self, hidden_dim=20):
        super(ProjectilePINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)  
        )

    def forward(self, t):
        return self.net(t)

    def physics_loss(self, t_interior, g=9.8):
        t_interior.requires_grad_(True)
        pred = self.forward(t_interior)
        x_pred = pred[:, 0:1]
        y_pred = pred[:, 1:2]
        dx_dt = torch.autograd.grad(x_pred, t_interior, grad_outputs=torch.ones_like(x_pred), create_graph=True)[0]
        dy_dt = torch.autograd.grad(y_pred, t_interior, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
        d2x_dt2 = torch.autograd.grad(dx_dt, t_interior, grad_outputs=torch.ones_like(dx_dt), create_graph=True)[0]
        d2y_dt2 = torch.autograd.grad(dy_dt, t_interior, grad_outputs=torch.ones_like(dy_dt), create_graph=True)[0]
        res_x = d2x_dt2
        res_y = d2y_dt2 + g
        return torch.mean(res_x**2 + res_y**2)

    def data_loss(self, t_data, x_data, y_data):
        pred = self.forward(t_data)
        x_pred = pred[:, 0:1]
        y_pred = pred[:, 1:2]
        return torch.mean((x_pred - x_data)**2 + (y_pred - y_data)**2)

    @staticmethod
    def train_model(t_data, x_data, y_data, T=2.0, g=9.8, n_phys_points=50, 
                    n_epochs=5000, lr=1e-3, early_stop_patience=500, early_stop_min_delta=1e-6):
        t_interior = torch.linspace(0, T, n_phys_points).reshape(-1, 1)
        model = ProjectilePINN(hidden_dim=20)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            p_loss = model.physics_loss(t_interior, g=g)
            d_loss = model.data_loss(t_data, x_data, y_data)
            loss_value = p_loss + d_loss
            loss_value.backward()
            optimizer.step()
            
            current_loss = loss_value.item()
            if current_loss < best_loss - early_stop_min_delta:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 500 == 0:
                print(f"[Projectile] Epoch {epoch:06d} | PDE Loss = {p_loss.item():.6f} | Data Loss = {d_loss.item():.6f} | Total = {current_loss:.6f}")

            if patience_counter >= early_stop_patience:
                print(f"[Projectile] Early stopping triggered at epoch {epoch} with best loss {best_loss:.6f}.")
                break
                
        return model

    @staticmethod
    def load_trajectories(base_directory, file_name, kernel_size=11, apply_smoothing=True):
        trajectories = []
        for traj_num in range(200):
            video_dir = os.path.join(base_directory, f"video_{traj_num}")
            file_path = os.path.join(video_dir, file_name)
            if not os.path.exists(file_path):
                continue
            try:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
            except Exception as e:
                print(f"[Projectile] Error loading trajectory {traj_num}: {e}")
                continue
            data_len = len(data)
            t = np.arange(data_len)
            x = np.array([item[0] if item is not None else np.nan for item in data])
            y = np.array([item[1] if item is not None else np.nan for item in data])
            valid_idx = np.where(~np.isnan(x))[0]
            if len(valid_idx) == 0:
                continue
            start = valid_idx[0]
            x = x[start:]
            y = y[start:]
            t = t[start:]
            df = pd.DataFrame({'t': t, 'x': x, 'y': y}).interpolate(method='linear')
            traj_x = -df['x'].to_numpy() / 100.0
            traj_y = -df['y'].to_numpy() / 100.0
            traj_t = np.linspace(0, 6, len(traj_x))
            t_data = torch.tensor(traj_t, dtype=torch.float32).reshape(-1, 1)
            x_data = torch.tensor(traj_x, dtype=torch.float32).reshape(-1, 1)
            y_data = torch.tensor(traj_y, dtype=torch.float32).reshape(-1, 1)
            if apply_smoothing:
                t_data, x_data, y_data = apply_filter(t_data, x_data, y_data, kernel_size=kernel_size)
            trajectories.append((t_data, x_data, y_data))
        return trajectories

    @staticmethod
    def evaluate(trajectories, output_dir):
        results = []
        mse_list, nmse_list = [], []
        os.makedirs(output_dir, exist_ok=True)

        for i, (t_data, x_data, y_data) in enumerate(trajectories):
            print(f"\n[Projectile] Training PINN for Trajectory {i+1}...")
            model = ProjectilePINN.train_model(
                t_data=t_data, x_data=x_data, y_data=y_data, T=2.0, g=9.8, n_phys_points=50,
                n_epochs=200000, lr=1e-3
            )
            t_test = torch.linspace(0, 6, 200).reshape(-1, 1)
            pred = model(t_test).detach().numpy()
            x_pred = pred[:, 0]
            y_pred = pred[:, 1]
            x_true = np.interp(t_test.numpy().flatten(), t_data.numpy().flatten(), x_data.numpy().flatten())
            y_true = np.interp(t_test.numpy().flatten(), t_data.numpy().flatten(), y_data.numpy().flatten())
            mse_xy = np.mean((x_pred - x_true)**2 + (y_pred - y_true)**2)
            nmse_xy = mse_xy / np.var(np.concatenate([x_true, y_true]))
            mse_list.append(mse_xy)
            nmse_list.append(nmse_xy)
            results.append({"Trajectory": i+1, "MSE": mse_xy, "NMSE": nmse_xy})
            print(f"Trajectory {i+1} Metrics: MSE: {mse_xy:.6f} | NMSE: {nmse_xy:.6f}")
            plt.figure(figsize=(8, 5))
            plt.subplot(2, 1, 1)
            plt.plot(t_test.numpy(), x_true, 'b-', label='True X', linewidth=2)
            plt.plot(t_test.numpy(), x_pred, 'r--', label='Pred X')
            plt.scatter(t_data.numpy(), x_data.numpy(), color='k', s=20, label='Data X')
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(t_test.numpy(), y_true, 'b-', label='True Y', linewidth=2)
            plt.plot(t_test.numpy(), y_pred, 'r--', label='Pred Y')
            plt.scatter(t_data.numpy(), y_data.numpy(), color='k', s=20, label='Data Y')
            plt.legend()
            plt.suptitle(f"Projectile Trajectory\nMSE: {mse_xy:.6f} | NMSE: {nmse_xy:.6f}", fontsize=12)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(output_dir, f"projectile_traj_{i+1}.png"))
            plt.close()

        avg_mse = np.mean(mse_list)
        avg_nmse = np.mean(nmse_list)
        results.append({"Trajectory": "Average", "MSE": avg_mse, "NMSE": avg_nmse})
        results_df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "projectile_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\n[Projectile] Average MSE: {avg_mse:.6f} | Average NMSE: {avg_nmse:.6f}")
        print(f"Results saved to {csv_path}")
        return avg_mse, avg_nmse

def run_pin_framework(base_directory, file_name="3D_centers.pkl", phenomenon="pendulum"):
    """
    Run PINN training for a chosen physical phenomenon.
    
    Parameters:
        base_directory (str): The base directory containing trajectory folders.
        file_name (str): Trajectory data filename. Default is "3D_centers.pkl".
        phenomenon (str): Physical phenomenon ("pendulum", "freefall", or "projectile").
    """
    phenomenon = phenomenon.lower()
    preprocess_directories(base_directory)

    base_name = os.path.basename(os.path.normpath(base_directory))
    output_dir = f"{base_name}_results"
    
    if phenomenon == "pendulum":
        print("[Framework] Running Pendulum PINN.")
        trajectories = PendulumPINN.load_trajectories(base_directory, file_name)
        trajectories = [(t, x - torch.mean(x)) for t, x in trajectories]
        PendulumPINN.evaluate(trajectories, output_dir)
    elif phenomenon == "freefall":
        print("[Framework] Running Free Fall PINN.")
        trajectories = FreeFallPINN.load_trajectories(base_directory, file_name)
        FreeFallPINN.evaluate(trajectories, output_dir)
    elif phenomenon == "projectile":
        print("[Framework] Running Projectile PINN.")
        trajectories = ProjectilePINN.load_trajectories(base_directory, file_name, kernel_size=11, apply_smoothing=True)
        ProjectilePINN.evaluate(trajectories, output_dir)
    else:
        print("Unknown phenomenon. Choose from: pendulum, freefall, projectile.")

if __name__ == "__main__":
    run_pin_framework("cogvideo_x_1-5_projectile_glm-4_i2v_prompt", file_name="3D_centers.pkl", phenomenon="projectile")

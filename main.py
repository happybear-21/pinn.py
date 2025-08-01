"""
PINN for Modeling Vertical Ball Trajectory under Gravity
========================================================
This script trains a Physics-Informed Neural Network (PINN) to predict
the vertical trajectory of a ball under gravity. The network learns from
noisy observations while being constrained by the known physics:
    dh/dt = v0 - g * t
and the initial condition:
    h(0) = h0

Author: [Your Name]
Date: 2025-08-01
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ---------------------- Configurable Constants ----------------------

g = 9.8          # gravitational acceleration (m/s^2)
h0 = 1.0         # initial height (meters)
v0 = 10.0        # initial velocity (m/s)

t_min, t_max = 0.0, 2.0
N_data = 10
noise_level = 0.7

n_hidden = 20
num_epochs = 4000
lr = 0.01
weight_decay = 1e-4

lambda_data = 0.1
lambda_ode = 10.0
lambda_ic = 10.0

print_every = 200

# ---------------------- Device Setup ----------------------

def get_device():
    """
    Returns the best available device: CUDA, MPS (Apple), or CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

# ---------------------- True Physics Function ----------------------

def true_solution(t):
    """
    Analytical solution of the vertical trajectory of a ball under gravity.
    """
    return h0 + v0 * t - 0.5 * g * (t ** 2)

# ---------------------- Data Generation ----------------------

def generate_noisy_data():
    """
    Generates synthetic noisy data based on the true physics.
    """
    np.random.seed(0)
    t_data = np.linspace(t_min, t_max, N_data)
    h_clean = true_solution(t_data)
    h_noisy = h_clean + noise_level * np.random.randn(N_data)
    return t_data, h_noisy

t_data, h_data_noisy = generate_noisy_data()

t_data_tensor = torch.tensor(t_data, dtype=torch.float32).view(-1, 1).to(device)
h_data_tensor = torch.tensor(h_data_noisy, dtype=torch.float32).view(-1, 1).to(device)

# ---------------------- PINN Model ----------------------

class PINN(nn.Module):
    """
    Physics-Informed Neural Network for modeling ball trajectory.
    """
    def __init__(self, n_hidden=20):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1)
        )
    
    def forward(self, t):
        return self.net(t)

model = PINN(n_hidden=n_hidden).to(device)

# ---------------------- Loss Functions ----------------------

def derivative(y, x):
    """
    Computes dy/dx using PyTorch autograd.
    """
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

def physics_loss(model, t):
    """
    Computes the loss enforcing the ODE: dh/dt = v0 - g * t
    """
    t_physics = t.clone().detach().requires_grad_(True)
    h_pred = model(t_physics)
    dh_dt = derivative(h_pred, t_physics)
    dh_dt_true = v0 - g * t_physics
    return torch.mean((dh_dt - dh_dt_true) ** 2)

def initial_condition_loss(model):
    """
    Penalizes deviation from initial condition: h(0) = h0
    """
    t0 = torch.zeros(1, 1, dtype=torch.float32).to(device)
    h0_pred = model(t0)
    return (h0_pred - h0).pow(2).mean()

def data_loss(model, t_data, h_data):
    """
    Mean squared error on noisy training data.
    """
    h_pred = model(t_data)
    return torch.mean((h_pred - h_data) ** 2)

# ---------------------- Training ----------------------

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    l_data = data_loss(model, t_data_tensor, h_data_tensor)
    l_ode = physics_loss(model, t_data_tensor)
    l_ic = initial_condition_loss(model)
    
    loss = lambda_data * l_data + lambda_ode * l_ode + lambda_ic * l_ic
    loss.backward()
    optimizer.step()

    if (epoch + 1) % print_every == 0:
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Total Loss: {loss.item():.6f} | "
              f"Data: {l_data.item():.6f} | ODE: {l_ode.item():.6f} | IC: {l_ic.item():.6f}")

# ---------------------- Evaluation ----------------------

model.eval()
t_plot = np.linspace(t_min, t_max, 100).reshape(-1, 1).astype(np.float32)
t_plot_tensor = torch.tensor(t_plot.tolist(), dtype=torch.float32, requires_grad=True).to(device)
h_pred_plot = model(t_plot_tensor).detach().cpu().tolist()
h_true_plot = true_solution(t_plot)

# ---------------------- Plotting ----------------------

plt.figure(figsize=(8, 5))
plt.scatter(t_data, h_data_noisy, color='red', label='Noisy Data')
plt.plot(t_plot, h_true_plot, 'k--', label='Exact Solution')
plt.plot(t_plot, h_pred_plot, 'b', label='PINN Prediction')
plt.xlabel('Time (t)')
plt.ylabel('Height h(t)')
plt.legend()
plt.title('PINN for Ball Trajectory under Gravity')
plt.grid(True)
plt.tight_layout()
plt.show()

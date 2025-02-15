import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from tqdm.auto import tqdm
from matplotlib.colors import LogNorm

# ----------------------------
# Parameters and Setup
# ----------------------------
N_FRAMES = 300  # total number of frames for one direction
kappas_forward = jnp.logspace(0, 2, N_FRAMES)  # condition numbers from 1 to 100
kappas_backward = jnp.flip(kappas_forward)  # reverse sequence
kappas = jnp.concatenate([kappas_forward, kappas_backward])  # combine both sequences
N_FRAMES = len(kappas)  # update total number of frames
FPS = 60
FILENAME = "chebyshev_gd.mp4"
DPI = 250
n_points = 200

# Update constants to match the style
LABELS_FONTSIZE = 10
FIGSIZE = (5, 5)
xlim = [-5, 5]
ylim = [-5, 5]

# Create a grid over the plotting domain
x_vals = np.linspace(xlim[0], xlim[1], n_points)
y_vals = np.linspace(ylim[0], ylim[1], n_points)
X, Y = np.meshgrid(x_vals, y_vals)

# Fixed rotation matrix (theta = 30°)
theta = np.radians(30)
Q = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

# Define starting point before plot setup
x0 = np.array([3.0, 3.0])

# ----------------------------
# Helper functions
# ----------------------------
def quadratic_function_on_grid(A, X, Y):
    """
    Compute the quadratic function f(x,y) = 0.5 * [x, y] A [x, y]^T
    on a grid given by X and Y.
    """
    # Since A is symmetric, we use:
    # f(x,y)=0.5*(A[0,0]*x^2 + 2*A[0,1]*x*y + A[1,1]*y^2)
    return 0.5 * (A[0, 0]*X**2 + 2*A[0, 1]*X*Y + A[1, 1]*Y**2)

def compute_gd_trajectory(A, x0, alpha, iterations=7):
    """
    Compute the gradient descent trajectory for the quadratic function
    f(x) = 0.5 * x^T A x starting from x0.
    Gradient is A*x (since b=0), and the update is:
        x_new = x - alpha * A * x
    Returns an array of points (including the starting point).
    """
    trajectory = [x0.copy()]
    x = x0.copy()
    for _ in range(iterations):
        grad = A @ x
        x = x - alpha * grad
        trajectory.append(x.copy())
    return np.array(trajectory)

def compute_accel_trajectory(A, x0, iterations=7, kappa=None):
    """
    Compute the accelerated gradient descent trajectory using Chebyshev polynomials.
    For the quadratic function f(x) = 0.5 * x^T A x starting from x0.
    """
    trajectory = [x0.copy()]
    x_prev = x0.copy()
    x_curr = x0.copy()
    
    # For our problem, μ=1 and L=kappa
    L = kappa  # Using the passed kappa value
    mu = 1.0
    
    # Handle the case when L = μ (condition number = 1)
    if abs(L - mu) < 1e-10:
        # For L = μ, the problem is perfectly conditioned
        # Simple gradient descent will converge in one step
        grad = A @ x_curr
        x_next = x_curr - (1/L) * grad
        trajectory.extend([x_next.copy()] * (iterations-1))
        return np.array(trajectory)
    
    # Compute x = (L+μ)/(L-μ) for Chebyshev evaluation
    x_cheb = (L + mu)/(L - mu)
    
    # First step is regular gradient descent
    grad = A @ x_curr
    alpha_0 = 2/(L + mu)
    x_next = x_curr - alpha_0 * grad
    trajectory.append(x_next.copy())
    
    x_prev = x_curr
    x_curr = x_next
    
    # Initialize t values
    t_prev = 1.0  # T_0(x) = 1
    t_curr = x_cheb  # T_1(x) = x
    
    # Subsequent steps use acceleration
    for k in range(1, iterations-1):
        # Use Chebyshev polynomial recurrence: T_{k+1}(x) = 2xT_k(x) - T_{k-1}(x)
        t_next = 2 * x_cheb * t_curr - t_prev
        
        # Compute step sizes
        beta_k = (t_prev) / t_next
        alpha_k = 4 / (L - mu) * t_curr/t_next
        
        # Compute gradient
        grad = A @ x_curr
        
        # Update position
        x_next = x_curr - alpha_k * grad + beta_k * (x_curr - x_prev)
        
        trajectory.append(x_next.copy())
        
        # Update variables for next iteration
        x_prev = x_curr
        x_curr = x_next
        t_prev = t_curr
        t_curr = t_next
    
    return np.array(trajectory)

# ----------------------------
# Set up the initial static plots for comparison
# ----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), layout='tight')

# Setup first subplot (κ=1)
kappa0 = float(kappas[0])
A0 = Q @ np.diag([1, kappa0]) @ Q.T
Z0 = quadratic_function_on_grid(A0, X, Y)

# Pre-compute fixed contour levels based on both min and max kappa
min_kappa = float(kappas_forward[0])
max_kappa = float(kappas_forward[-1])

# Get Z values for both extremes
A_min = Q @ np.diag([1, min_kappa]) @ Q.T
A_max = Q @ np.diag([1, max_kappa]) @ Q.T
Z_min = quadratic_function_on_grid(A_min, X, Y)
Z_max = quadratic_function_on_grid(A_max, X, Y)

# Use global min and max for contour levels with logarithmic spacing
global_min = Z_min.min() # Avoid zero or negative values
global_max = max(Z_min.max(), Z_max.max())

contour_levels = np.logspace(np.log10(global_min), np.log10(global_max), 15)
# Create LogNorm that matches the actual data range
norm = LogNorm(1e2*global_min, 1e3*global_max)

contour1 = ax1.contour(X, Y, Z0, levels=contour_levels, cmap="BuPu", norm=norm)
ax1.set_ylim(ylim)
ax1.set_xlim(xlim)
ax1.set_aspect('equal')
ax1.grid(alpha=0.4, linestyle=":")
ax1.set_xlabel(r"$x_1$", fontsize=LABELS_FONTSIZE)
ax1.set_ylabel(r"$x_2$", fontsize=LABELS_FONTSIZE)

# Plot GD trajectory for κ=1
alpha0 = 2/(1 + kappa0)
traj0_gd = compute_gd_trajectory(A0, x0, alpha0, iterations=7)
ax1.plot(traj0_gd[:, 0], traj0_gd[:, 1], 'r-o', lw=1.5, markersize=3, label='GD', zorder=10)

# Plot accelerated trajectory for κ=1
traj0_accel = compute_accel_trajectory(A0, x0, iterations=7, kappa=kappa0)
ax1.plot(traj0_accel[:, 0], traj0_accel[:, 1], 'b-o', lw=1.5, markersize=3, label='Accelerated GD', zorder=10)

ax1.plot(x0[0], x0[1], 'ko', markersize=5, zorder=11)
ax1.plot(0, 0, marker='*', color='gold', markersize=10, zorder=11)
ax1.set_title(f"$\\varkappa={kappa0:5.1f}$", fontsize=LABELS_FONTSIZE)
ax1.legend(fontsize=8)

# Setup second subplot (κ=100)
kappa_max = float(kappas_forward[-1])
A_max = Q @ np.diag([1, kappa_max]) @ Q.T
Z_max = quadratic_function_on_grid(A_max, X, Y)

contour2 = ax2.contour(X, Y, Z_max, levels=contour_levels, cmap="BuPu", norm=norm)
ax2.set_ylim(ylim)
ax2.set_xlim(xlim)
ax2.set_aspect('equal')
ax2.grid(alpha=0.4, linestyle=":")
ax2.set_xlabel(r"$x_1$", fontsize=LABELS_FONTSIZE)
ax2.set_ylabel(r"$x_2$", fontsize=LABELS_FONTSIZE)

# Plot GD trajectory for κ=100
alpha_max = 2/(1 + kappa_max)
traj_max_gd = compute_gd_trajectory(A_max, x0, alpha_max, iterations=7)
ax2.plot(traj_max_gd[:, 0], traj_max_gd[:, 1], 'r-o', lw=1.5, markersize=3, label='GD', zorder=10)

# Plot accelerated trajectory for κ=100
traj_max_accel = compute_accel_trajectory(A_max, x0, iterations=7, kappa=kappa_max)
ax2.plot(traj_max_accel[:, 0], traj_max_accel[:, 1], 'b-o', lw=1.5, markersize=3, label='Accelerated GD ', zorder=10)

ax2.plot(x0[0], x0[1], 'ko', markersize=5, zorder=11)
ax2.plot(0, 0, marker='*', color='gold', markersize=10, zorder=11)
ax2.set_title(f"$\\varkappa={kappa_max:5.1f}$", fontsize=LABELS_FONTSIZE)
ax2.legend(fontsize=8)

# Save the combined static figure
plt.savefig('condition_number_comparison.pdf',
            dpi=DPI,
            bbox_inches='tight',
            format='pdf')

# Clear the current figure before proceeding with animation
plt.close()

# ----------------------------
# Create new figure for animation
# ----------------------------
fig, ax = plt.subplots(figsize=FIGSIZE, layout='tight')

# Initial setup for animation
kappa0 = float(kappas[0])
A0 = Q @ np.diag([1, kappa0]) @ Q.T
Z0 = quadratic_function_on_grid(A0, X, Y)

# Initial contour plot
contour = ax.contour(X, Y, Z0, levels=contour_levels, cmap="BuPu", norm=norm)
ax.set_ylim(ylim)
ax.set_xlim(xlim)
ax.set_aspect('equal')
ax.grid(alpha=0.4, linestyle=":")
ax.set_xlabel(r"$x_1$", fontsize=LABELS_FONTSIZE)
ax.set_ylabel(r"$x_2$", fontsize=LABELS_FONTSIZE)

# Initial GD trajectory plot
alpha0 = 2/(1 + kappa0)
traj0_gd = compute_gd_trajectory(A0, x0, alpha0, iterations=7)
line_traj_gd, = ax.plot(traj0_gd[:, 0], traj0_gd[:, 1], 'r-o', lw=1.5, markersize=3, 
                          label='GD', zorder=10)

# Initial accelerated trajectory plot
traj0_accel = compute_accel_trajectory(A0, x0, iterations=7, kappa=kappa0)
line_traj_accel, = ax.plot(traj0_accel[:, 0], traj0_accel[:, 1], 'b-o', lw=1.5, markersize=3, 
                           label='Accelerated GD', zorder=10)

# Add markers
ax.plot(x0[0], x0[1], 'ko', markersize=5, zorder=11)
ax.plot(0, 0, marker='*', color='gold', markersize=10, zorder=11)
title = ax.set_title(f"$\\varkappa={kappa0:5.1f}$", fontsize=LABELS_FONTSIZE)

# Add legend with better positioning
ax.legend(fontsize=8, bbox_to_anchor=(0.8 , -0.1), frameon=False, ncol=2)


# Add social media handle
ax.text(1, -0.12, '@fminxyz', 
        transform=ax.transAxes,
        fontsize=12, 
        color='grey', 
        alpha=0.7,
        ha='right',
        va='bottom')

# Animation update function
def update(frame):
    kappa = float(kappas[frame])
    A = Q @ np.diag([1, kappa]) @ Q.T
    Z = quadratic_function_on_grid(A, X, Y)
    
    # Clear previous contours
    for coll in ax.collections[:]:
        coll.remove()
    
    # Update contours with fixed levels
    ax.contour(X, Y, Z, levels=contour_levels, cmap="BuPu", norm=norm)
    
    # Update GD trajectory
    alpha = 2/(1 + kappa)
    traj_gd = compute_gd_trajectory(A, x0, alpha, iterations=7)
    line_traj_gd.set_data(traj_gd[:, 0], traj_gd[:, 1])
    
    # Update accelerated trajectory
    traj_accel = compute_accel_trajectory(A, x0, iterations=7, kappa=kappa)
    line_traj_accel.set_data(traj_accel[:, 0], traj_accel[:, 1])
    
    # Update title with fixed width format
    title.set_text(f"$\\varkappa={kappa:5.1f}$")
    
    return line_traj_gd, line_traj_accel, title

# ----------------------------
# Create and run the animation
# ----------------------------
print("Generating animation...")
ani = FuncAnimation(fig, update, frames=N_FRAMES, interval=100, blit=True)

# Optimize video writing with tight layout
writer = animation.FFMpegWriter(
    fps=FPS,
    metadata=dict(artist='@fminxyz'),
    bitrate=-1,
    codec='h264',
    extra_args=['-preset', 'ultrafast',  # Faster encoding
                '-crf', '28',  # Constant Rate Factor (18-28 is good, lower=better quality)
                '-pix_fmt', 'yuv420p', # Compatible pixel format
                '-tune', 'animation']  
)

with tqdm(total=100, desc="Saving animation") as pbar:
    plt.tight_layout()
    ani.save(FILENAME, 
            writer=writer,
            dpi=DPI,
            progress_callback=lambda i, n: pbar.update(100/n))

plt.close()

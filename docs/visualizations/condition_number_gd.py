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
FILENAME = "condition_number_gd_light.mp4"
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

# ----------------------------
# Set up the initial plot
# ----------------------------
fig, ax = plt.subplots(figsize=FIGSIZE, layout='tight')

# Initial condition: use the first kappa value
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

# Initial plot with fixed levels
contour = ax.contour(X, Y, Z0, levels=contour_levels, cmap="BuPu", norm=norm)
ax.set_ylim(ylim)
ax.set_xlim(xlim)
ax.set_aspect('equal')
ax.grid(alpha=0.4, linestyle=":")
ax.set_xlabel(r"$x_1$", fontsize=LABELS_FONTSIZE)
ax.set_ylabel(r"$x_2$", fontsize=LABELS_FONTSIZE)

# Compute and plot initial trajectory with higher zorder
x0 = np.array([3.0, 3.0])
alpha0 = 2 / (1 + kappa0)
traj0 = compute_gd_trajectory(A0, x0, alpha0, iterations=7)
line_traj, = ax.plot(traj0[:, 0], traj0[:, 1], 'r-o', lw=1.5, markersize=3, 
                     label='GD trajectory', zorder=10)

# Update markers style with even higher zorder
start_point, = ax.plot(x0[0], x0[1], 'ko', markersize=5, label='Start', zorder=11)
min_point, = ax.plot(0, 0, marker='*', color='gold',markersize=10, label='Minimum', zorder=11)
title = ax.set_title(f"$\\varkappa=\\mbox{{\\texttt{{{kappa0:5.1f}}}}}$", fontsize=LABELS_FONTSIZE)

# Add social media handle in bottom right corner
ax.text(1, -0.12, '@fminxyz', 
        transform=ax.transAxes,  # Use axes coordinates (0-1)
        fontsize=12, 
        color='grey', 
        alpha=0.7,
        ha='right',  # Horizontal alignment: right
        va='bottom'  # Vertical alignment: bottom
        )

# ----------------------------
# Animation update function
# ----------------------------
def update(frame):
    kappa = float(kappas[frame])
    A = Q @ np.diag([1, kappa]) @ Q.T
    Z = quadratic_function_on_grid(A, X, Y)
    
    # Clear previous contours
    for coll in ax.collections[:]:
        coll.remove()
    
    # Update contours with fixed levels
    contour = ax.contour(X, Y, Z, levels=contour_levels, cmap="BuPu", norm=norm)
    
    # Update trajectory
    alpha = 2 / (1 + kappa)
    traj = compute_gd_trajectory(A, x0, alpha, iterations=7)
    line_traj.set_data(traj[:, 0], traj[:, 1])
    
    # Update title with fixed width format
    title.set_text(f"$\\varkappa={kappa:5.1f}$")
    
    return line_traj, title

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
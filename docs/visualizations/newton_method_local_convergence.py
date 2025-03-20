import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import grad
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from tqdm.auto import tqdm

# Основная функция (по минимуму из двух — основной и вспомогательной частей)
def f(x):
    return jnp.where(x <= -1, (x - 1)**2,
           jnp.where(x >= 1, (x + 1)**2,
                     -0.25 * x**4 + 2.5 * x**2 + 1.75))

def df(x):
    return grad(f)(x)

def ddf(x):
    return grad(df)(x)

# Вспомогательные части функции для пунктирной отрисовки (то, что было бы на левой и правой частях)
def f_left(x):
    return (x - 1)**2  # центральная часть, продолженная влево

def f_right(x):
    return (x + 1)**2 # центральная часть, продолженная вправо

# Реализация метода Ньютона
def newton_step(x):
    return x - df(x) / ddf(x)

# Compute Newton trajectory for a given starting point
def compute_newton_trajectory(x0, iterations=5):
    x_points = [x0]
    
    for i in range(iterations):
        x_next = newton_step(x_points[-1])
        x_points.append(x_next)
    
    return x_points

# ----------------------------
# Animation parameters
# ----------------------------
N_FRAMES = 1500  # total number of frames
FPS = 60
FILENAME = "newton_method_local_convergence.mp4"
DPI = 250
iterations = 5  # Number of Newton iterations per frame

# Generate uniformly spaced starting points from -1.4 to 1.4
starting_points = np.linspace(-1.45, 1.45, int(N_FRAMES/2))
starting_points = np.concatenate([starting_points, starting_points[::-1]])
# Setup figure
fig, ax = plt.subplots(figsize=(7, 4), layout='tight')

# x values for plotting the function
x_vals = np.linspace(-1.5, 1.5, 500)
x_left_dashed = x_vals
x_right_dashed = x_vals

# y values for the function and dashed parts
y_main = np.array([f(float(x)).item() for x in x_vals])
y_left_dashed = np.array([f_left(x) for x in x_left_dashed])
y_right_dashed = np.array([f_right(x) for x in x_right_dashed])

# Initial plot of the function
main_line, = ax.plot(x_vals, y_main, color='black', linewidth=3)  # основная функция
left_dashed, = ax.plot(x_left_dashed, y_left_dashed, linestyle='--', color='gray', linewidth=2)
right_dashed, = ax.plot(x_right_dashed, y_right_dashed, linestyle='--', color='gray', linewidth=2)

# Initial trajectory (will be updated in the animation)
initial_points = compute_newton_trajectory(starting_points[0], iterations)
initial_y_values = [f(x).item() for x in initial_points]

# Plot trajectory line and points
traj_line, = ax.plot([], [], 'r--', linewidth=1.5)
traj_points, = ax.plot([], [], 'ro', markersize=6)

# Set plot limits and styling
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([0, 6])
ax.grid(linestyle=":")
title = ax.set_title(f'Newton Method: Starting point x₀ = {starting_points[0]:.2f}')

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
    x0 = starting_points[frame]
    
    # Compute Newton trajectory for this starting point
    x_points = compute_newton_trajectory(x0, iterations)
    y_points = [f(x).item() for x in x_points]
    
    # Update trajectory line and points
    traj_line.set_data(x_points, y_points)
    traj_points.set_data(x_points, y_points)
    
    # Update title
    title.set_text(r'$x_{k+1} = x_k - \left[\nabla^2 f(x_k)\right]^{-1} \nabla f(x_k)$, $x_0 = $' + f'{x0:.2f}')
    
    return traj_line, traj_points, title

# Create and run the animation
print("Generating animation...")
ani = FuncAnimation(fig, update, frames=N_FRAMES, interval=1000/FPS, blit=True)

# Configure video writer
writer = animation.FFMpegWriter(
    fps=FPS,
    metadata=dict(artist='Newton Method Animation'),
    bitrate=-1,
    codec='h264',
    extra_args=['-preset', 'ultrafast',
                '-crf', '28',
                '-pix_fmt', 'yuv420p',
                '-tune', 'animation']  
)

# Save the animation
with tqdm(total=100, desc="Saving animation") as pbar:
    plt.tight_layout()
    ani.save(FILENAME, 
            writer=writer,
            dpi=DPI,
            progress_callback=lambda i, n: pbar.update(100/n))

plt.close()
print(f"Animation saved as {FILENAME}")
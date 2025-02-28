import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from tqdm.auto import tqdm
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------
# Parameters and Setup
# ----------------------------
N_FRAMES = 120  # total number of frames
FPS = 20
FILENAME = "qr_algorithm_3.mp4"
DPI = 200
MATRIX_SIZE = 128  # size of the matrices

# Style parameters
LABELS_FONTSIZE = 10
FIGSIZE = (12, 4)

# Generate three different types of matrices
np.random.seed(42)  # for reproducibility

# 1. Random matrix
A1 = np.random.randn(MATRIX_SIZE, MATRIX_SIZE)

# 2. Random symmetric matrix
temp = np.random.randn(MATRIX_SIZE, MATRIX_SIZE)
A2 = (temp + temp.T) / 2

# 3. Random tridiagonal matrix (symmetric version)
main_diag = np.random.randn(MATRIX_SIZE)
off_diag = np.random.randn(MATRIX_SIZE - 1)
A3 = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

# Store initial matrices for title display
A1_init = A1.copy()
A2_init = A2.copy()
A3_init = A3.copy()

# Precompute mask for enforcing tridiagonality: elements with |i-j| > 1 will be zeroed
tridiag_mask = np.abs(np.subtract.outer(np.arange(MATRIX_SIZE), np.arange(MATRIX_SIZE))) > 1

def qr_iteration(A):
    """Perform one step of QR iteration"""
    Q, R = np.linalg.qr(A)
    return R @ Q

# ----------------------------
# Set up the plot
# ----------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=FIGSIZE, layout='tight')

# Initial heatmaps
im1 = ax1.imshow(A1, cmap='twilight')
im2 = ax2.imshow(A2, cmap='twilight')
im3 = ax3.imshow(A3, cmap='twilight')

# Set titles
ax1.set_title('Random Matrix', fontsize=LABELS_FONTSIZE)
ax2.set_title('Symmetric Matrix', fontsize=LABELS_FONTSIZE)
ax3.set_title('Tridiagonal Matrix', fontsize=LABELS_FONTSIZE)
plt.tight_layout()

# Remove ticks
for ax in [ax1, ax2, ax3]:
    ax.set_xticks([])
    ax.set_yticks([])

# Add social media handle
fig.text(0.975, 0.95, '@fminxyz', fontsize=12, color='grey', alpha=0.7,
         ha='right', va='bottom')

# Animation update function
def update(frame):
    global A1, A2, A3
    A1 = qr_iteration(A1)
    A2 = qr_iteration(A2)
    A3 = qr_iteration(A3)
    # Enforce tridiagonality for A3 by zeroing out all elements where |i-j| > 1
    # A3[tridiag_mask] = 0
    
    im1.set_array(A1)
    im2.set_array(A2)
    im3.set_array(A3)
    return im1, im2, im3

# ----------------------------
# Create and save the animation
# ----------------------------
print("Generating animation...")
ani = FuncAnimation(fig, update, frames=N_FRAMES, interval=100, blit=True)

# Optimize video writing
writer = animation.FFMpegWriter(
    fps=FPS,
    metadata=dict(artist='@fminxyz'),
    bitrate=-1,
    codec='h264',
    extra_args=['-preset', 'ultrafast', '-crf', '28', '-pix_fmt', 'yuv420p', '-tune', 'animation']
)

with tqdm(total=100, desc="Saving animation") as pbar:
    plt.tight_layout()
    ani.save(FILENAME, writer=writer, dpi=DPI,
             progress_callback=lambda i, n: pbar.update(100/n))

plt.close()

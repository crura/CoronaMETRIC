import git, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from functions import JS_Div
from scipy.stats import entropy

repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir

# Configuration
frames = 40
amplitude = 0.40
x = np.linspace(-5, 5, 1000)
mu_A, sigma = 0, 1.5
p = amplitude * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu_A) / sigma) ** 2)

# Set up plot
fig, ax = plt.subplots(figsize=(8, 5))
line_p, = ax.plot([], [], label="Distribution A (p)", linewidth=2)
line_q, = ax.plot([], [], label="Distribution B (q)", linewidth=2, linestyle='--')
text_kl = ax.text(-4.5, 0.14, '', fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))

ax.set_xlim(-5, 5)
ax.set_ylim(0, 0.15)
ax.set_title("KL Divergence as Distribution B Aligns with A")
ax.set_xlabel("x")
ax.set_ylabel("Scaled Probability Density")
ax.grid(True)
ax.legend()

# Animation function
def animate(i):
    mu_shift = -2 + (4 * i / (frames - 1))  # shift from -2 to 2
    q = amplitude * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu_shift) / sigma) ** 2)
    kl = entropy(p, q)

    line_p.set_data(x, p)
    line_q.set_data(x, q)
    text_kl.set_text(f'Total KL(p || q) = {kl:.4f}')

    return line_p, line_q, text_kl

# Run animation
ani = FuncAnimation(fig, animate, frames=frames, interval=100, blit=True)

# Save to GIF
ani.save(os.path.join(repo_path,"Output/kl_divergence_shift2.gif"), writer=PillowWriter(fps=10))






import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Define x-axis and distributions
x = np.linspace(-5, 5, 1000)
mu_p, sigma_p = 0, 1.5
mu_q, sigma_q = 1, 1.5
amp = 0.40

# Gaussian distributions p(x) and q(x)
p = amp * (1 / (sigma_p * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu_p) / sigma_p) ** 2)
q = amp * (1 / (sigma_q * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu_q) / sigma_q) ** 2)

# Total KL Divergence
kl_total = entropy(p, q)
js_total = JS_Div(p, q)

# Points for annotation
x1 = -3.0     # p > q
x2 = 2       # p < q
x3 = 0.5000000053122178   # p ≈ q

# Function to evaluate p, q, and KL term at a given x
def evaluate_kl_point(x_point):
    p_val = amp * (1 / (sigma_p * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_point - mu_p) / sigma_p) ** 2)
    q_val = amp * (1 / (sigma_q * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_point - mu_q) / sigma_q) ** 2)
    kl_val = p_val * np.log(p_val / q_val)
    return p_val, q_val, kl_val

# Evaluate the three points
p1, q1, kl1 = evaluate_kl_point(x1)
p2, q2, kl2 = evaluate_kl_point(x2)
p3, q3, kl3 = evaluate_kl_point(x3)

# Plot setup
plt.figure(figsize=(10, 8))
plt.plot(x, p, label="Distribution A (p)", color='orange', linewidth=2)
plt.plot(x, q, label="Distribution B (q)", color='orangered', linestyle='--', linewidth=2)

# Annotations
plt.plot(x1, p1, 'ro')
plt.annotate(f'p(x) > q(x)\nKL > 0\n({kl1:.2e})',
             xy=(x1, p1), xytext=(x1 - 2, p1 + 0.05),
             arrowprops=dict(facecolor='darkred', shrink=0.05),
             color='darkred', fontsize=10)

plt.plot(x2, p2, 'bo')
plt.annotate(f'p(x) < q(x)\nKL < 0\n({kl2:.2e})',
             xy=(x2, p2), xytext=(x2 + 0.5, p2 + 0.05),
             arrowprops=dict(facecolor='navy', shrink=0.05),
             color='navy', fontsize=10)

plt.plot(x3, p3, 'go')
plt.annotate(f'p(x) = q(x)\nKL = 0\n({kl3:.2e})',
             xy=(x3, p3), xytext=(x3 - 2, p3 - 0.08),
             arrowprops=dict(facecolor='darkgreen', shrink=0.05),
             color='darkgreen', fontsize=10, verticalalignment='top')


# KL total text inside the axis
plt.text(-4.5, 0.1, f'Total KL(p || q) = {kl_total:.4f}',
         fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))

# Final styling
plt.title("KL Divergence Illustration for Two Distributions")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(repo_path,"Output/kl_divergence_example_fig.png"))




# 1. Compute the difference between p and q
diff = p - q

# 2. Find where the sign of the difference changes
sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]

# 3. Interpolate the x-values where sign changes occur
intersections = []
for idx in sign_changes:
    # x0 and x1 are around the sign change
    x0, x1 = x[idx], x[idx + 1]
    y0, y1 = diff[idx], diff[idx + 1]
    
    # Linear interpolation for root (intersection point)
    x_inter = x0 - y0 * (x1 - x0) / (y1 - y0)
    intersections.append(x_inter)

print("Intersection points (p ≈ q):", intersections)



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Define x-axis and distributions
x = np.linspace(-5, 5, 1000)
mu_p, sigma_p = 0, 1.5
mu_q, sigma_q = 1, 1.5
amp = 0.40

# Gaussian distributions p(x) and q(x)
p = amp * (1 / (sigma_p * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu_p) / sigma_p) ** 2)
q = amp * (1 / (sigma_q * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu_q) / sigma_q) ** 2)

# Jensen-Shannon average distribution
m = 0.5 * (p + q)

# Total KL and JS Divergence
kl_total = entropy(p, q)
kl_p_m = entropy(p, m)
kl_q_m = entropy(q, m)
jsd = 0.5 * (kl_p_m + kl_q_m)

# Example x-values
x1 = -1.5
x2 = 2
x3 = 0.5

# Function to evaluate p, q, m at a given x
def eval_dist(x_val):
    p_val = amp * (1 / (sigma_p * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_val - mu_p) / sigma_p) ** 2)
    q_val = amp * (1 / (sigma_q * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_val - mu_q) / sigma_q) ** 2)
    m_val = 0.5 * (p_val + q_val)
    return p_val, q_val, m_val

# Evaluate values at the three example points
p1, q1, m1_y = eval_dist(x1)
p2, q2, m2_y = eval_dist(x2)
p3, q3, m3_y = eval_dist(x3)

# --- Plotting ---
plt.figure(figsize=(10, 8))

# Plot the distributions
plt.plot(x, p, label="Distribution A (p)", color='orange', linewidth=2)
plt.plot(x, q, label="Distribution B (q)", color='orangered', linestyle='--', linewidth=2)
plt.plot(x, m, label="Mixture m(x) = ½(p + q)", color='purple', linestyle=':', linewidth=2)

# Plot all example points (p, q, m) with their respective colors
example_points = [(x1, p1, 'orange'), (x1, q1, 'orangered'), (x1, m1_y, 'purple'),
                  (x2, p2, 'orange'), (x2, q2, 'orangered'), (x2, m2_y, 'purple'),
                  (x3, p3, 'orange'), (x3, q3, 'orangered'), (x3, m3_y, 'purple')]

for x_val, y_val, color in example_points:
    plt.plot(x_val, y_val, 'o', color=color, markersize=6)

# Title and divergence summary text
plt.title("Jensen–Shannon Divergence Example", pad=20)

plt.text(-4, 0.10, f'Total KL(p || q) = {kl_total:.4f}',
         fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))
plt.text(-4, 0.09, f'Total JS(p || q) = {jsd:.4f}',
         fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))

# Final styling
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(repo_path,"Output/js_divergence_example_fig.png"))

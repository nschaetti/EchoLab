import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Paramètres physiques
# -----------------------------
g = 9.81      # gravité (m/s²)
L = 1.0       # longueur du fil (m)
m = 1.0       # masse (kg)
b = 0.1       # coefficient de frottement

# -----------------------------
# Discrétisation temporelle
# -----------------------------
dt = 0.01
t = np.arange(0, 15, dt)

# -----------------------------
# État initial
# -----------------------------
theta = np.zeros_like(t, dtype=float)
theta[0] = np.pi / 4       # 45°

# On impose une vitesse angulaire initiale nulle et on calcule la valeur au pas suivant
# avec un développement limité à l'ordre 2 pour démarrer le schéma centré.
theta_dot0 = 0.0
theta_ddot0 = -(b / m) * theta_dot0 - (g / L) * np.sin(theta[0])
theta[1] = theta[0] + dt * theta_dot0 + 0.5 * dt**2 * theta_ddot0

# -----------------------------
# Boucle temporelle (différences finies explicites)
# -----------------------------
for n in range(1, len(t) - 1):
    theta[n + 1] = (
        2 * theta[n] - theta[n - 1]
        - (b * dt / m) * (theta[n] - theta[n - 1])
        - (dt**2) * (g / L) * np.sin(theta[n])
    )
# end for

# Vitesse angulaire estimée par différentiation numérique
theta_dot = np.gradient(theta, dt)

# -----------------------------
# Animation du pendule
# -----------------------------
x = L * np.sin(theta)
y = -L * np.cos(theta)

fig, (ax_phase, ax_pend) = plt.subplots(1, 2, figsize=(10, 5))

# Sous-plot phase (theta, theta_dot)
ax_phase.plot(theta, theta_dot, color="silver", linewidth=1.5)
theta_margin = 0.1 * (theta.max() - theta.min()) or 0.1
theta_dot_margin = 0.1 * (theta_dot.max() - theta_dot.min()) or 0.1
ax_phase.set_xlim(theta.min() - theta_margin, theta.max() + theta_margin)
ax_phase.set_ylim(theta_dot.min() - theta_dot_margin, theta_dot.max() + theta_dot_margin)
ax_phase.set_xlabel(r"$\theta$ (rad)")
ax_phase.set_ylabel(r"$\dot{\theta}$ (rad/s)")
ax_phase.set_title("Espace de phase")
(phase_point,) = ax_phase.plot([], [], "o", color="crimson")

# Sous-plot du pendule
ax_pend.set_xlim(-1.1 * L, 1.1 * L)
ax_pend.set_ylim(-1.1 * L, 0.1 * L)
ax_pend.set_aspect('equal', adjustable='box')
ax_pend.set_title("Pendule amorti – Méthode des différences finies explicites")

# Élément graphique du pendule
(line,) = ax_pend.plot([], [], lw=2, color="royalblue")
(bob,) = ax_pend.plot([], [], "o", color="darkorange", markersize=12)

# Fonction d'initialisation
def init():
    line.set_data([], [])
    bob.set_data([], [])
    phase_point.set_data([], [])
    return line, bob, phase_point
# end def init

# Fonction de mise à jour
def update(frame):
    line.set_data([0, x[frame]], [0, y[frame]])
    bob.set_data([x[frame]], [y[frame]])
    phase_point.set_data([theta[frame]], [theta_dot[frame]])
    return line, bob, phase_point
# end def update

# Animation
ani = FuncAnimation(
    fig, update, frames=len(t), init_func=init,
    interval=20, blit=True
)

plt.show()

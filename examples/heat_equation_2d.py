import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Paramètres physiques
D = 0.0001        # diffusivité
nx, ny = 100, 100
dx = 1.0
dt = 0.1
nt = 2000
HEAT_SOURCE = {
    "intensity": 10.0,   # chaleur injectée (°C/s), négatif pour un puits
    "center": None,      # (y, x), None => centre du domaine
    "width": 10,
    "height": None,      # None => même valeur que width
}

INITIAL_CONDITION = {
    "mode": "line",       # options : point, cloud, square, line
    "temperature": 100.0,  # température maximale visée
    "background": 0.0,     # température du milieu
    "center": None,        # None => centre du domaine
    "cloud": {
        "num_blobs": 5,
        "sigma": 6.0,
        "seed": 42,
        "amplitude_jitter": 0.25,
    },
    "square": {
        "width": 20,
        "height": None,  # None => même valeur que width
    },
    "line": {
        "orientation": "horizontal",  # horizontal ou vertical
        "thickness": 3,
        "offset": 0,  # décalage par rapport au centre
    },
}


def create_initial_temperature(nx, ny, config):
    """Construit un champ de température initial à partir d'un générateur choisi."""
    mode = config.get("mode", "point")
    max_temp = float(config.get("temperature", 100.0))
    background = float(config.get("background", 0.0))
    center = config.get("center")
    if center is None:
        center = (ny // 2, nx // 2)
    cy = int(np.clip(center[0], 0, ny - 1))
    cx = int(np.clip(center[1], 0, nx - 1))

    T = np.full((ny, nx), background, dtype=float)

    if mode == "point":
        T[cy, cx] = max_temp
        return T

    if mode == "square":
        square_cfg = config.get("square", {})
        width = max(1, int(square_cfg.get("width", 20)))
        height = square_cfg.get("height")
        height = max(1, int(height if height is not None else width))
        y_start = cy - height // 2
        x_start = cx - width // 2
        y_end = y_start + height
        x_end = x_start + width
        if y_start < 0:
            y_start = 0
            y_end = min(height, ny)
        if x_start < 0:
            x_start = 0
            x_end = min(width, nx)
        if y_end > ny:
            y_end = ny
            y_start = max(0, ny - height)
        if x_end > nx:
            x_end = nx
            x_start = max(0, nx - width)
        T[y_start:y_end, x_start:x_end] = max_temp
        return T

    if mode == "line":
        line_cfg = config.get("line", {})
        orientation = line_cfg.get("orientation", "horizontal").lower()
        thickness = max(1, int(line_cfg.get("thickness", 1)))
        offset = int(line_cfg.get("offset", 0))
        if orientation.startswith("h"):
            y_center = int(np.clip(cy + offset, 0, ny - 1))
            y_start = y_center - thickness // 2
            y_end = y_start + thickness
            if y_start < 0:
                y_start = 0
                y_end = min(thickness, ny)
            if y_end > ny:
                y_end = ny
                y_start = max(0, ny - thickness)
            T[y_start:y_end, :] = max_temp
        elif orientation.startswith("v"):
            x_center = int(np.clip(cx + offset, 0, nx - 1))
            x_start = x_center - thickness // 2
            x_end = x_start + thickness
            if x_start < 0:
                x_start = 0
                x_end = min(thickness, nx)
            if x_end > nx:
                x_end = nx
                x_start = max(0, nx - thickness)
            T[:, x_start:x_end] = max_temp
        else:
            raise ValueError(f"Orientation de ligne inconnue : {orientation}")
        return T

    if mode == "cloud":
        cloud_cfg = config.get("cloud", {})
        num_blobs = max(1, int(cloud_cfg.get("num_blobs", 5)))
        sigma = float(cloud_cfg.get("sigma", 6.0))
        jitter = float(cloud_cfg.get("amplitude_jitter", 0.0))
        seed = cloud_cfg.get("seed")
        rng = np.random.default_rng(seed)
        y_indices, x_indices = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        for _ in range(num_blobs):
            yc = rng.uniform(0, ny - 1)
            xc = rng.uniform(0, nx - 1)
            amplitude = max_temp * (1 + jitter * rng.uniform(-1.0, 1.0))
            gaussian = np.exp(-(((x_indices - xc) ** 2) + ((y_indices - yc) ** 2)) / (2 * sigma ** 2))
            T += amplitude * gaussian
        return T

    raise ValueError(f"Mode de condition initiale inconnu : {mode}")


def create_heat_source_mask(nx, ny, config):
    """Construit un masque booléen pour appliquer une source carrée constante."""
    intensity = float(config.get("intensity", 0.0))
    if intensity == 0.0:
        return None, 0.0

    center = config.get("center")
    if center is None:
        center = (ny // 2, nx // 2)
    cy = int(np.clip(center[0], 0, ny - 1))
    cx = int(np.clip(center[1], 0, nx - 1))

    width = max(1, int(config.get("width", 1)))
    height = config.get("height")
    height = max(1, int(height if height is not None else width))

    y_start = cy - height // 2
    x_start = cx - width // 2
    y_end = y_start + height
    x_end = x_start + width

    if y_start < 0:
        y_start = 0
        y_end = min(height, ny)
    if x_start < 0:
        x_start = 0
        x_end = min(width, nx)
    if y_end > ny:
        y_end = ny
        y_start = max(0, ny - height)
    if x_end > nx:
        x_end = nx
        x_start = max(0, nx - width)

    mask = np.zeros((ny, nx), dtype=bool)
    mask[y_start:y_end, x_start:x_end] = True
    return mask, intensity


T = create_initial_temperature(nx, ny, INITIAL_CONDITION)
vmax = max(float(INITIAL_CONDITION.get("temperature", 100.0)), np.max(T))
heat_source_mask, heat_source_intensity = create_heat_source_mask(nx, ny, HEAT_SOURCE)

# Pour la stabilité (condition de Courant)
assert D*dt/dx**2 < 0.25, "Instable : réduire dt ou augmenter dx"

# Préparation de l'animation
fig, ax = plt.subplots()
im = ax.imshow(T, cmap='hot', origin='lower', vmin=0, vmax=vmax if vmax > 0 else 1.0)
cbar = fig.colorbar(im, ax=ax, label='Température')
title = ax.set_title("t = 0.00 s")


def update(frame):
    # Calcule un pas d'Euler explicite pour la diffusion thermique
    T_new = T.copy()
    T_new[1:-1, 1:-1] = (
        T[1:-1, 1:-1]
        + D * dt / dx**2
        * (
            T[2:, 1:-1]
            + T[:-2, 1:-1]
            + T[1:-1, 2:]
            + T[1:-1, :-2]
            - 4 * T[1:-1, 1:-1]
        )
    )
    if heat_source_mask is not None:
        T_new[heat_source_mask] += heat_source_intensity * dt
    T[:] = T_new
    im.set_data(T)
    current_min = np.min(T)
    current_max = np.max(T)
    vmin, vmax_current = im.get_clim()
    if current_min < vmin or current_max > vmax_current:
        im.set_clim(min(current_min, 0.0), max(current_max, 1e-12))
    title.set_text(f"t = {(frame + 1) * dt:.2f} s")
    return (im,)


animation = FuncAnimation(fig, update, frames=nt, interval=50, blit=False)

plt.show()

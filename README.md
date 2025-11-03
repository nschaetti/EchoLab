# EchoLab

EchoLab is a Python package for seismic modeling and simulation.

## Features

- Velocity model generation and manipulation
- Acoustic wavefield simulation
- Seismic data processing and visualization

## Velocity Models

EchoLab provides a set of classes for handling velocity models in different dimensions:

- `VelocityModel1D`: For 1D velocity models (depth-dependent)
- `VelocityModel2D`: For 2D velocity models (cross-sections with x and z dimensions)
- `VelocityModel3D`: For 3D velocity models (x, y, and z dimensions)

These classes provide methods for:

- Accessing velocity data in different formats (numpy array, list, torch tensor)
- Getting metadata about the model (dx, dz, max, min, dimensions)
- Serialization to different formats (numpy, pickle, json)

### Example Usage

```python
from echolab.modeling import VelocityModel2D

# Create a 2D velocity model
velocity_data = ... # 2D numpy array with shape (nz, nx)
dx, dz = 10.0, 5.0
model = VelocityModel2D(velocity_data, (dz, dx))

# Access properties
print(f"Shape: {model.shape}")
print(f"Grid spacing: {model.grid_spacing}")
print(f"Min velocity: {model.min_velocity}")
print(f"Max velocity: {model.max_velocity}")

# Access data in different formats
numpy_data = model.as_numpy()
list_data = model.as_list()
torch_data = model.as_torch()  # Requires PyTorch

# Save to file
model.save("velocity_model.npz", format="numpy")
model.save("velocity_model.pkl", format="pickle")
model.save("velocity_model.json", format="json")

# Load from file
from echolab.modeling import load_velocity_model
model = load_velocity_model("velocity_model.npz")
```

## Simulation

EchoLab provides simulators for acoustic wavefield propagation:

- `OpenFWISimulator`: For forward modeling
- `OpenANFWISimulator`: For ambient noise modeling

These simulators can use the VelocityModel classes for input.

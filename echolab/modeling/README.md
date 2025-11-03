# EchoLab Velocity Models

This directory contains the implementation of velocity models for the EchoLab package.

## Pydantic Implementation

The velocity model classes have been updated to use [pydantic](https://docs.pydantic.dev/) for data validation and management. This provides several benefits:

1. **Automatic validation**: Input data is validated when creating models
2. **Type checking**: Ensures that data has the correct types
3. **Better error messages**: Clear error messages when validation fails
4. **Serialization/deserialization**: Easy conversion to/from JSON and other formats

## Main Classes

### VelocityMap

`VelocityMap` is a pydantic model that represents a velocity model with specific dimensionality. It stores the velocity data and grid spacing information.

```python
from echolab.modeling.velocity_models import VelocityMap, Dimensionality

# Create a 1D velocity map
data_1d = np.linspace(1500, 3000, 100).astype(np.float32)
vmap_1d = VelocityMap.from_1d_array(data_1d, dz=10.0)

# Create a 2D velocity map
data_2d = np.ones((50, 100), dtype=np.float32) * 2000
vmap_2d = VelocityMap.from_2d_array(data_2d, dx=5.0, dz=10.0)

# Create a 3D velocity map
data_3d = np.ones((20, 30, 40), dtype=np.float32) * 2500
vmap_3d = VelocityMap.from_3d_array(data_3d, dx=5.0, dy=7.5, dz=10.0)
```

### VelocityModel

`VelocityModel` is the base class for velocity models. It uses `VelocityMap` internally to store and manage velocity data.

There are three concrete implementations:

1. `VelocityModel1D`: For 1D velocity models
2. `VelocityModel2D`: For 2D velocity models
3. `VelocityModel3D`: For 3D velocity models

```python
from echolab.modeling.velocity_models import VelocityModel1D, VelocityModel2D, VelocityModel3D

# Create a 1D velocity model
data_1d = np.linspace(1500, 3000, 100).astype(np.float32)
model_1d = VelocityModel1D.from_array(data_1d, grid_spacing=10.0)

# Create a 2D velocity model
data_2d = np.ones((50, 100), dtype=np.float32) * 2000
model_2d = VelocityModel2D.from_array(data_2d, grid_spacing=(5.0, 10.0))

# Create a 3D velocity model
data_3d = np.ones((20, 30, 40), dtype=np.float32) * 2500
model_3d = VelocityModel3D.from_array(data_3d, grid_spacing=(5.0, 7.5, 10.0))
```

## Serialization and Deserialization

Velocity models can be saved to and loaded from files in various formats:

```python
# Save a model
model.save(path, format="numpy")  # Options: "numpy", "json", "pickle"

# Load a model
from echolab.modeling.velocity_models import load_velocity_model
model = load_velocity_model(path)

# Save multiple models
from echolab.modeling.velocity_models import save_velocity_models
save_velocity_models([model1, model2, model3], path)

# Load multiple models
from echolab.modeling.velocity_models import load_velocity_models
models = load_velocity_models(path)
```

## Backward Compatibility

The new implementation maintains backward compatibility with the original implementation. The main entry point is the `velocity_models.py` module, which re-exports the pydantic-based implementations with the same interface as the original classes.

```python
from echolab.modeling.velocity_models import (
    VelocityMap,
    Dimensionality,
    VelocityModel,
    VelocityModel1D,
    VelocityModel2D,
    VelocityModel3D,
    load_velocity_model,
    save_velocity_models,
    load_velocity_models,
    save_velocity_maps,
    load_velocity_maps
)
```
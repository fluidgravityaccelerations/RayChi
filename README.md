# RayChi
RayChi: A Taichi-Powered Ray Tracer

![Rendering Example](https://github.com/fluidgravityaccelerations/RayChi/blob/main/raychiRenderingExample.png) 
*(Above is a RayChi rendering example)*

A GPU-accelerated ray tracer built with Taichi Lang. Supports ambient occlusion, direct lighting, reflective/refractive materials, and configurable scenes.

## Features

- 🚀 GPU-accelerated rendering using Taichi
- 🌟 Multiple material types (diffuse, metal, dielectric, emissive)
- 💡 Direct lighting and ambient occlusion
- 🖼 Configurable scene setup via JSON
- 📦 CLI and Python API interfaces

## Installation

### Prerequisites
- Python 3.7+
- [Taichi](https://taichi-lang.org/) (will be installed automatically)

### Install from source
```bash
git clone https://github.com/fluidgravityaccelerations/RayChi.git
cd RayChi
pip install .

# AR/VR Cloth Cutting Simulation

A physics-based cloth cutting system with real-time mesh manipulation and realistic cloth behavior.

## Quick Start

```powershell
python cloth_simulation.py
```

Or double-click `run.bat`

## Files

- `cloth_simulation.py` - Main simulation code
- `requirements.txt` - Python dependencies
- `run.bat` - One-click launcher

## Installation

```powershell
pip install -r requirements.txt
```

Or install manually:
```powershell
pip install numpy PyOpenGL PyOpenGL-accelerate
```

## Controls
- **Left Mouse + Drag** - Cut the cloth
- **Right Mouse + Drag** - Rotate camera
- **SPACE** - Horizontal cut
- **C** - Random cut
- **R** - Reset cloth
## Features

 Multi-layer triangulated cloth mesh  
 Verlet integration physics  
 Real-time cutting with line intersection  
 Spring constraint system  
 Interactive 3D visualization  

## How It Works

- **Physics**: Verlet integration with spring constraints
- **Cutting**: Line-triangle intersection detection
- **Rendering**: OpenGL 3D visualization
- **Cloth**: Particle system with structural, shear, and layer springs

---


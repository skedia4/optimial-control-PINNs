# ME598-SciML-Project

## Optimal Control Using PINNs

### Establish goals
1. Define a linear dynamical system for trajectory optimization
2. Derive the governing PDE for the system
3. Train the PDE for optimal control loss using terminal cost as boundary and residual loss for PDE
4. Compare against analytical closed form solution using Ricatti equation
5. Create the framework for Linear/non-linear program in CaSADi and perform trajectory optimization
5. Compare the accuracy against the closed form solution and computation time
6. Repeat for non-linear system
7. Repeat for higher dimension system
8. Encode environmental constraints(such as obstacles) to trajectory optimization
8. Show the pipeline implemented on a robotic simulation platform

### Identify your Data
1. Data will be generated from PDE directly or computed from known closed form solution

### Anticipate storage needs
1. No specific storage limitations. Data can be stored locally or in the box folder.

### Commit to a naming scheme
1. Each experiment has it's own base code eg. "Linear_System_3D" and shared helper functions.

### Outline your verfication plan
1. The verification is always attempted against closed form solution. The choice of the dynamical system is crucial so that solution is always available.

### Map out your computing resources
1. For low dimension system training no extra computing resources are required
2. For higher dimension system, we may need a GPU cluster to train the network
3. Trajectory optimization using linear/non-linear program in CaSADi can also be implemeted in personal laptop

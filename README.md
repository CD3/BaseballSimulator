# About

`BaseballSimulator` is a python module for simulating the flight of a baseball with spin. It was written for an undergraduate
research project in the Physics Department at Fort Hays State University (in Hays, KS). The project aims to answer the question
of why a curveball appears to "break". The first step was to implement a physics simulation to generate baseball flight
paths, which resulted in `BaseballSimulator`.

It is still being developed to test new ideas.

## Authors

The physics implemented in `BaseballSimulator` was written by two undergraduate students, June Jung and Richard Whitehill. The
original code was rewritten to provide a cleaner API by C.D. Clark III. 

# Installing

`BaseballSimulator` is available on PyPi.

```bash
$ pip install baseballsimulator
```

# Usage

`BaseballSimulator` provides a library for running projectile motion simulations. The simulator is not specific to baseball, but
by default, it is configured to simulate a baseball. The simulation includes both the drag force, and the Magnus force, which is the force
produced by spin. The equations of motion can be written

![equations%20of%20motion](http://latex.codecogs.com/gif.latex?\vec{F}%20=%20m\vec{a}%20=%20-g\hat{z}%20-%20\alpha%20v\vec{v}%20+%20\beta%20\vec{v}\times\vec{\omega})

To use the simulator, import it from `BaseballSimulator`, configure the launch, and run.

```python
# import the simulator
from BaseballSimulator.Simulator import *


# create a simulation
simulation = Simulation()
# turn off drag in the simulation by multiplying the drag coefficent by zero
simulation.config.drag_coefficient *= 0

# configure the launch.
# projectile released 1 m above the ground at x-y origin
# launched at 100 mph, in the +y direction, at an angle of 45 degree
launch_config = LaunchConfiguration()
launch_config.position = Q_(1,'m')*zhat
launch_config.speed = Q_(100,'mph')
launch_config.direction = [0,1,1]

# run the simulation until the projectile hits
# the ground (the z-component of its position becomes negative)
trajectory = simulation.run(launch_config, terminate_function = lambda current_trajectory: current_trajectory[-1][3] < 0)


# look at the initial and final state
print("initial state:"    , trajectory[0])
print("initial time:"     , trajectory[0][0])
print("initial position:" , trajectory[0][1:4])
print("initial velocity:" , trajectory[0][4:7])
print("initial spin:"     , trajectory[0][7:10])
print()
print("final state:"      , trajectory[-1])
print("final time:"       , trajectory[-1][0])
print("final position:"   , trajectory[-1][1:4])
print("final velocity:"   , trajectory[-1][4:7])
print("final spin:"       , trajectory[-1][7:10])


# turn drag back on and rerun
simulation.config.drag_coefficient = Q_(0.000788, 'kg/m')
trajectory = simulation.run(launch_config, terminate_function = lambda current_trajectory: current_trajectory[-1][3] < 0)

print("initial state:"    , trajectory[0])
print("initial time:"     , trajectory[0][0])
print("initial position:" , trajectory[0][1:4])
print("initial velocity:" , trajectory[0][4:7])
print("initial spin:"     , trajectory[0][7:10])
print()
print("final state:"      , trajectory[-1])
print("final time:"       , trajectory[-1][0])
print("final position:"   , trajectory[-1][1:4])
print("final velocity:"   , trajectory[-1][4:7])
print("final spin:"       , trajectory[-1][7:10])

```
A couple of things to note.
Simulations are ran using a `Simulation` instance. This is where the simulation physics
is configured, and it expects configuration parameters to be specified as [Pint Quantities](https://pint.readthedocs.io/en/0.10.1/). `Q_` is an alias for Pint's `Quantiyt` class. The unit registry is automatically setup by the library, so you
do not need to import `pint` directly.

Specific launch configurations are configured with a `LaunchConfiguration` instance. This is where the initial state
of the projectile (position, speed, spin) are set. This class also expects Pint quantities. The three
Cartesian unit vectors (`xhat`, `yhat`, `zyat`) are defined for convenience. In the example above, the initial
position is set to 1 m above ground by multiplying `Q_(1,'m')` by the unit vector that points in the
+z direction.

To run a simulation, we call the `run` method of the `Simulator` instance and pass it a `LaunchConfiguration` instance.
We can also pass a "terminate function". This function will be called after each iteration in the simulation,
and will be passed the current record of all states computed during the simulation. It should return true when the
simulation it to terminate, and false otherwise. In the example above, we want to run the simulation until the projectile
hits the ground, so we test that the z component of the position vector in the last state computed is greater than zero.
The default terminate function runs 1000 iterations and exits.

When the simulation finishes, it returns a "trajectory", which is just a list of all of the projectile states that
where computed. The projectile state is represented as a `pyTorch` tensor, and contains the time, position, velocity,
and spin.

Running the example above will output the following:
```
initial state: tensor([ 0.0000,  0.0000,  0.0000,  1.0000,  0.0000, 31.6105, 31.6105, -0.0000,
        -0.0000, -0.0000])
initial time: tensor(0.)
initial position: tensor([0., 0., 1.])
initial velocity: tensor([ 0.0000, 31.6105, 31.6105])
initial spin: tensor([-0., -0., -0.])

final state: tensor([ 6.4848e+00,  0.0000e+00,  2.0499e+02, -2.7792e-02,  0.0000e+00,
         3.1611e+01, -3.1932e+01,  0.0000e+00,  0.0000e+00,  0.0000e+00])
final time: tensor(6.4848)
final position: tensor([ 0.0000e+00,  2.0499e+02, -2.7792e-02])
final velocity: tensor([  0.0000,  31.6105, -31.9324])
final spin: tensor([0., 0., 0.])
initial state: tensor([ 0.0000,  0.0000,  0.0000,  1.0000,  0.0000, 31.6105, 31.6105, -0.0000,
        -0.0000, -0.0000])
initial time: tensor(0.)
initial position: tensor([0., 0., 1.])
initial velocity: tensor([ 0.0000, 31.6105, 31.6105])
initial spin: tensor([-0., -0., -0.])

final state: tensor([ 5.4664e+00,  0.0000e+00,  1.1607e+02, -1.0568e-02,  0.0000e+00,
         1.4574e+01, -2.3785e+01,  0.0000e+00,  0.0000e+00,  0.0000e+00])
final time: tensor(5.4664)
final position: tensor([ 0.0000e+00,  1.1607e+02, -1.0568e-02])
final velocity: tensor([  0.0000,  14.5745, -23.7848])
final spin: tensor([0., 0., 0.])
```

Note the final position for the two simulations. Without drag, the ball travels 205 meter (all state parameters
are expressed in base SI units). With drag, this is 116 meter. This indicates that a 380 foot fly ball could
travel 672 feet without air resistance.

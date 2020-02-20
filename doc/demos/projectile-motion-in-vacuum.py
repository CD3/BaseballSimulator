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


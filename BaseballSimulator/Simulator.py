

# standard lib modules
import types
import pprint
import numbers

# third-party modules
import pint
import torch
import numpy
import scipy
from scipy.spatial.transform import Rotation


# module configuration
ureg = pint.UnitRegistry()
ureg.define('percent = 0.01 rad')
Q_ = ureg.Quantity
pint.set_application_registry(ureg)

ScalarType = torch.float

def Rx(theta):
  return torch.from_numpy( Rotation.from_rotvec(torch.tensor([theta.to('radian').magnitude,0,0])).as_dcm() ).type(ScalarType)
def Ry(theta):
  return torch.from_numpy( Rotation.from_rotvec(torch.tensor([0,theta.to('radian').magnitude,0])).as_dcm() ).type(ScalarType)
def Rz(theta):
  return torch.from_numpy( Rotation.from_rotvec(torch.tensor([0,0,theta.to('radian').magnitude])).as_dcm() ).type(ScalarType)

Norm = numpy.linalg.norm

xhat = torch.tensor([1,0,0],dtype=ScalarType)
yhat = torch.tensor([0,1,0],dtype=ScalarType)
zhat = torch.tensor([0,0,1],dtype=ScalarType)

def num(quant):
  '''
  Return numerical value of quantity expressed in base units (SI)
  '''
  return quant.to_base_units().magnitude


class Simulation:
  def __init__(self):
    # default configuration
    self.config = types.SimpleNamespace()
    self.config.wind_speed = Q_(0,'mph')
    self.config.wind_direction = Q_(0,'degree')
    self.config.drag_coefficient = Q_(0.0007884037809624002, 'kg/m')
    self.config.magnus_coefficient = Q_(2.2075e-06, 'kg * s / m')
    self.config.magnus_model = 'squared velocity'
    self.config.ball_mass = Q_(145,'g')
    self.config.ball_diameter  = Q_(3,'in')
    self.config.gravitational_acceleration = Q_(9.8,'m/s**2')
    self.config.time_step = Q_(1,'ms')
    self.config.time_step_growth_rate = Q_(1,'')
    self.config.error_tolerance = Q_(1,'percent')
    self.config.auto_converge_time_step = True

  def configure(self, config):
    config_keys_used = []
    for k in self.config.__dict__:
      config_key = None
      if k in config:
        config_key = k
      if k.replace("_"," ") in config:
        config_key = k.replace("_"," ")

      if config_key is not None:
        config_keys_used.append(config_key)
        new_val = Q_(config[config_key])
        if new_val.dimensionality != self.config.__dict__[k].dimensionality:
          raise Exception(f"Configuration parameter '{config_key}' has the wrong dimensions. Expected '{self.config.__dict__[k].dimensionality}' but got '{new_val.dimensionality}'.")
        self.config.__dict__[k] = new_val

    if len(config_keys_used) != len(config.keys()):
      print("Warning: there were unused keys when configuring Simulation:")
      for k in list( set(config.keys()) - set(config_keys_used) ):
        print("  ",k)
      print("Make sure you didn't mispell something.")

    





  @property
  def state_size(self):
    '''
    return the size of the state vector.

    current state vector entries:

    [0] : t
    [1:4] : x,y,z
    [4:7] : vx,vy,vz
    [7:10] : wx,wy,wz
    '''
    return 10





  def derivative(self, state):
      '''
      Calculate derivative of state vector.
      '''

      # Note:
      # state[0] == t
      # state[1:4] == [x,y,z]
      # state[4,7] == [vx,vy,vz]
      # state[7,10] == [wx,wy,wz]

      dsdt = torch.zeros( [self.state_size], dtype=state.dtype )

      dsdt[0] = 1                         # dt/dt = 1
      dsdt[1:4] = state[4:7]              # dx/dt = v
      dsdt[7:10] = torch.tensor([0,0,0],dtype=state.dtype)  # dw/dt = 0 ... for now


      # TODO: add wind
      # dv/dt = a = F/m = 1/m (Fg + Fd + Fm)
      # gravity
      dsdt[4:7] -= num(self.config.gravitational_acceleration)*zhat
      # drag
      speed = numpy.linalg.norm(state[4:7])
      dsdt[4:7] -= num(self.config.drag_coefficient)*speed*state[4:7] / num(self.config.ball_mass)
      # magnus
      if self.config.magnus_model == 'squared velocity':
        dsdt[4:7] += num(self.config.magnus_coefficient)*speed*torch.cross(state[7:10],state[4:7]) / num(self.config.ball_mass)
      elif self.config.magnus_model == 'linear velocity':
        dsdt[4:7] += num(self.config.magnus_coefficient)*torch.cross(state[7:10],state[4:7]) / num(self.config.ball_mass)
      else:
        raise f"Error: Unrecognized magnus model '{self.config.magnus_model}'"

      return dsdt


  def rk4(self, time_step, state):
      '''
      RK-4 method. Compute new state vector for a time in the future.
      '''

      # Runge-Kutta: k1 through k4
      k1 = time_step*self.derivative(state)
      k2 = time_step*self.derivative(state + k1*time_step/2)
      k3 = time_step*self.derivative(state + k2*time_step/2)
      k4 = time_step*self.derivative(state + k3*time_step  )

      # new accel, vel, pos, and t
      new_state = state + (k1 + 2*k2 + 2*k3 + k4)/6
      return new_state

  def _compute_error(self, s0, s1, s2):
    '''
    Compute the error between states s1 and s2 that will be used to determine if the time-step
    needs to be reduced. Both s1 and s2 are approximations to the same state, which follows state s0.
    
    s2 is the "better" approximation
    '''
    # compute percent difference  between states
    err = Norm(s2 - s1) / Norm(s2-s0)
    return err

  def run(self, launch_config, terminate_function = lambda record : len(record) > 1000, record_all=True):
    '''
    Takes a launch configuration, runs a simulation, and returns the trajectory.

    Trajectory is a list of state tensors. I.e, given

    trajecotry = sim.run(launch_config)

    trajectory will be a list. trajectory[0] will be the initial state of the projectile, trajectory[-1] will be the final state.

    Each state tensor contains 10 elements

    state[0]    contains time
    state[1:4]  contains cartesian coordinates
    state[4:7]  contains cartesian components of velocity vector
    state[7:10] contains cartesian components of spin (angular velocity) vector

    State quantities are expressed in SI BASE UNITS.
    '''

    state = torch.zeros( [self.state_size] )
    state[0] = 0
    state[1:4] = launch_config.get_position_tensor()
    state[4:7] = launch_config.get_velocity_tensor()
    state[7:10] = launch_config.get_spin_tensor()

    record = list()
    record.append(state.clone().detach())
    dt = self.config.time_step.to("s").magnitude
    while not terminate_function(record):

      while True:
        # take one long step and two short steps
        s1 = self.rk4( dt, state )
        s2 = self.rk4( dt/2, self.rk4( dt/2, state ) )
        err = self._compute_error(state,s1,s2)
        if self.config.auto_converge_time_step and err > self.config.error_tolerance.to("").magnitude:
          print(f"Info: decreasing time step from {dt} to {dt/2}")
          dt /= 2
        else:
          break

      state = s2

      if record_all:
        record.append( state.clone().detach() )
      else:
        record[0] = state.clone().detach()

      dt *= self.config.time_step_growth_rate.to("").magnitude


    return record



class LaunchConfiguration:
  def __init__(self):
    self.position = Q_(0,'ft')*xhat
    self.speed = Q_(0,'mph')
    self.direction = -yhat
    self.spin = Q_(0,'rpm')
    self.spin_direction = -xhat

  def configure(self, config):
    def load_tensor(v):
      if isinstance(v,list):
        if isinstance(v[0],Q_):
          # a list of quantities.
          # turn it into a tensor quantity.
          units = v[0].units
          return units*torch.tensor([ q.to(units).magnitude for q in v ],dtype=ScalarType)
        if isinstance(v[0],str):
          # a list of strings. treat as a list of quantities.
          # turn it into a tensor quantity.
          units = Q_(v[0]).units
          return units*torch.tensor([ Q_(q).to(units).magnitude for q in v ],dtype=ScalarType)
        if isinstance(v[0],numbers.Number):
          # a list of numbers.
          # turn it into a tensor.
          return torch.tensor(v,dtype=ScalarType)
      return v
    config_keys_used = []

    # handle vector (tensor) quantities first
    if 'position' in config:
      new_value = load_tensor(config['position'])
      self.position = new_value
      config_keys_used.append('position')

    if 'direction' in config:
      new_value = load_tensor(config['direction'])
      if isinstance(new_value,Q_):
        new_value = new_value.magnitude
      self.direction = new_value
      config_keys_used.append('direction')

    for k in ['spin_direction', 'spin direction']:
      if k in config:
        new_value = load_tensor(config[k])
        if isinstance(new_value,Q_):
          new_value = new_value.magnitude
        self.spin_direction = new_value
        config_keys_used.append(k)

    # now scalars
    for k in ['speed','spin']:
      if k in config:
        new_value = Q_(config[k])
        setattr(self,k,new_value)
        config_keys_used.append(k)





    if len(config_keys_used) != len(config.keys()):
      print("Warning: there were unused keys when configuring Simulation:")
      for k in list( set(config.keys()) - set(config_keys_used) ):
        print("  ",k)
      print("Make sure you didn't mispell something.")


  @property
  def velocity(self):
    uhat = self.direction / numpy.linalg.norm(self.direction)
    return self.speed*uhat

  @property
  def angular_velocity(self):
    uhat = self.spin_direction / numpy.linalg.norm(self.spin_direction)
    return self.spin*uhat


  def translate_position(self,dr):
    self.position += dr

  def point_velocity_at_position(self,r):
    self.direction = r - self.position
    self.direction = self.direction.magnitude

  def deflect_direction(self, R):
    self.direction = torch.mv(R,self.direction)

  def point_spin_at_position(self,r):
    self.spin_direction = r - self.position
    self.spin_direction = self.spin_direction.magnitude

  def deflect_spin_direction(self, R):
    self.spin_direction = torch.mv(R,self.spin_direction)

  def get_position_tensor(self,unit=ureg.meter):
    x = torch.tensor( [x.to(unit).magnitude for x in self.position],dtype=ScalarType )
    return x
  def get_velocity_tensor(self,unit=(ureg.meter/ureg.second)):
    v = torch.tensor( [v.to(unit).magnitude for v in self.velocity],dtype=ScalarType )
    return v
  def get_spin_tensor(self,unit=ureg.radian/ureg.second):
    w = torch.tensor( [w.to(unit).magnitude for w in self.angular_velocity],dtype=ScalarType )
    return w




if __name__ == '__main__':
  print("YES")

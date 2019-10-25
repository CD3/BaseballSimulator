

from BaseballSimulator import *

class Approx(object):
  def __init__(self,val):
    self._val = val
    self._epsilon = 0.01
  def epsilon(self,epsilon):
    self._epsilon = epsilon
    return self
  def __eq__(self,other):
    return abs(other - self._val) <= self._epsilon*abs(other + self._val)/2
  def __repr__(self):
    return "{} +/- {}%".format(self._val,self._epsilon*100)



def test_works():
  simulation = Simulation()
  launch_config = LaunchConfiguration()
  # trajectory = simulation.run(launch_config)




def test_launch_configuration():

  launch_config = LaunchConfiguration()

  launch_config.translate_position( Q_(55,'ft')*yhat )
  launch_config.translate_position( Q_(-1.5,'ft')*xhat )
  launch_config.translate_position( Q_(5.5,'ft')*zhat )

  x = launch_config.get_position_tensor()
  assert x[0] == Q_(-1.5,'ft').to("m").magnitude
  assert x[1] == Q_(55,'ft').to("m").magnitude
  assert x[2] == Q_(5.5,'ft').to("m").magnitude

  launch_config.speed = Q_(100,'mph')
  launch_config.point_velocity_at_position( Q_(5,'ft')*zhat )

  v = launch_config.get_velocity_tensor()
  L = Norm( launch_config.position.to("ft").magnitude ) 
  assert Approx(v[0]) ==  Q_(100,"mph").to("m/s").magnitude * 1.5 / L
  assert Approx(v[1]) ==  Q_(100,"mph").to("m/s").magnitude * -55 / L
  assert Approx(v[2]) ==  Q_(100,"mph").to("m/s").magnitude * -0.5 / L


  launch_config.spin = Q_(2000,'rpm')
  launch_config.point_spin_at_position( Q_(5,'ft')*zhat )

  w = launch_config.get_spin_tensor()
  L = Norm( launch_config.position.to("ft").magnitude ) 
  assert Approx(w[0]) ==  Q_(2000,"rpm").to("rad/s").magnitude * 1.5 / L
  assert Approx(w[1]) ==  Q_(2000,"rpm").to("rad/s").magnitude * -55 / L
  assert Approx(w[2]) ==  Q_(2000,"rpm").to("rad/s").magnitude * -0.5 / L


def test_launch_configuration_rotations():
  launch_config = LaunchConfiguration()

  launch_config.position = Q_(55,'ft')*yhat + Q_(-1.5,'ft')*xhat + Q_(5.5,'ft')*zhat
  launch_config.speed = Q_(100,'ft')
  launch_config.direction = -yhat
  launch_config.spin = Q_(2000,'rpm')
  launch_config.spin_direction = xhat


  assert Approx( Norm(launch_config.velocity.magnitude) ) == 100
  assert Approx( Norm(launch_config.angular_velocity.magnitude) ) == 2000

  assert Approx( launch_config.velocity.magnitude[0] ) == 0
  assert Approx( launch_config.velocity.magnitude[1] ) == -100
  assert Approx( launch_config.velocity.magnitude[2] ) == 0

  assert Approx( launch_config.angular_velocity.magnitude[0] ) == 2000
  assert Approx( launch_config.angular_velocity.magnitude[1] ) == 0
  assert Approx( launch_config.angular_velocity.magnitude[2] ) == 0


  launch_config.deflect_velocity( Rx(Q_(45,'degree')) )
  launch_config.deflect_spin( Rx(Q_(45,'degree')) )


  assert Approx( Norm(launch_config.velocity.magnitude) ) == 100
  assert Approx( Norm(launch_config.angular_velocity.magnitude) ) == 2000

  assert Approx( launch_config.velocity.magnitude[0] ) == 0
  assert Approx( launch_config.velocity.magnitude[1] ) == -100*2**0.5/2
  assert Approx( launch_config.velocity.magnitude[2] ) == -100*2**0.5/2

  assert Approx( launch_config.angular_velocity.magnitude[0] ) == 2000
  assert Approx( launch_config.angular_velocity.magnitude[1] ) == 0
  assert Approx( launch_config.angular_velocity.magnitude[2] ) == 0



  launch_config.deflect_velocity( Ry(Q_(90,'degree')) )
  launch_config.deflect_spin( Ry(Q_(90,'degree')) )


  assert Approx( Norm(launch_config.velocity.magnitude) ) == 100
  assert Approx( Norm(launch_config.angular_velocity.magnitude) ) == 2000

  assert Approx( launch_config.velocity.magnitude[0] ) == -100*2**0.5/2
  assert Approx( launch_config.velocity.magnitude[1] ) == -100*2**0.5/2
  assert       ( launch_config.velocity.magnitude[2] ) < 0.001

  assert       ( launch_config.angular_velocity.magnitude[0] ) < 0.001
  assert       ( launch_config.angular_velocity.magnitude[1] ) < 0.001
  assert Approx( launch_config.angular_velocity.magnitude[2] ) == -2000



def test_freefall_in_vacuum():
  simulation = Simulation()
  simulation.config.time_step = Q_(1,'s')
  simulation.config.drag_coefficient *= 0
  simulation.config.magnus_coefficient *= 0
  launch_config = LaunchConfiguration()
  launch_config.position = Q_(10,'m')*zhat
  trajectory = simulation.run(launch_config,lambda record: record[-1][3] < 0)

  dt = trajectory[-1][0]
  assert trajectory[-1][1] > -0.001
  assert trajectory[-1][1] < 0.001
  assert trajectory[-1][2] > -0.001
  assert trajectory[-1][2] < 0.001
  assert trajectory[-1][3] < 0
  assert trajectory[-1][4] > -0.001
  assert trajectory[-1][4] <  0.001
  assert trajectory[-1][5] > -0.001
  assert trajectory[-1][5] <  0.001
  assert Approx(trajectory[-1][6]) == -9.8*dt

def test_drag_only():
  simulation = Simulation()
  simulation.config.time_step = Q_(1,'s')
  simulation.config.gravitational_acceleration *= 0
  simulation.config.magnus_coefficient *= 0
  launch_config = LaunchConfiguration()
  launch_config.speed = Q_(10,'m/s')

  launch_config.direction = zhat
  trajectory = simulation.run(launch_config,lambda record: record[-1][0] > 1.0)

  alpha = simulation.config.drag_coefficient.to('kg/m').magnitude
  mass  = simulation.config.ball_mass.to('kg').magnitude

  dt = trajectory[-1][0]
  assert trajectory[-1][1] > -0.001
  assert trajectory[-1][1] < 0.001
  assert trajectory[-1][2] > -0.001
  assert trajectory[-1][2] < 0.001
  assert trajectory[-1][3] > 0

  assert trajectory[-1][4] > -0.001
  assert trajectory[-1][4] <  0.001
  assert trajectory[-1][5] > -0.001
  assert trajectory[-1][5] <  0.001
  assert trajectory[-1][6] >  0

  assert Approx(trajectory[-1][6]) == 10 / (1 + 10*(alpha/mass)*dt)
  assert Approx(trajectory[-1][3]) == (mass/alpha)*numpy.log( 1 + (10*alpha/mass)*dt)



  launch_config.direction = xhat
  trajectory = simulation.run(launch_config,lambda record: record[-1][0] > 1.0)

  dt = trajectory[-1][0]
  assert trajectory[-1][1] > 0
  assert trajectory[-1][2] > -0.001
  assert trajectory[-1][2] < 0.001
  assert trajectory[-1][3] > -0.001
  assert trajectory[-1][3] < 0.001

  assert trajectory[-1][4] >  0
  assert trajectory[-1][5] > -0.001
  assert trajectory[-1][5] <  0.001
  assert trajectory[-1][6] > -0.001
  assert trajectory[-1][6] <  0.001

  assert Approx(trajectory[-1][4]) == 10 / (1 + 10*(alpha/mass)*dt)
  assert Approx(trajectory[-1][1]) == (mass/alpha)*numpy.log( 1 + (10*alpha/mass)*dt)


  launch_config.direction = -yhat
  trajectory = simulation.run(launch_config,lambda record: record[-1][0] > 1.0)

  dt = trajectory[-1][0]
  assert trajectory[-1][1] > -0.001
  assert trajectory[-1][1] < 0.001
  assert trajectory[-1][2] < 0
  assert trajectory[-1][3] > -0.001
  assert trajectory[-1][3] < 0.001

  assert trajectory[-1][4] > -0.001
  assert trajectory[-1][4] <  0.001
  assert trajectory[-1][5] <  0
  assert trajectory[-1][6] > -0.001
  assert trajectory[-1][6] <  0.001

  assert Approx(trajectory[-1][5]) == -10 / (1 + 10*(alpha/mass)*dt)
  assert Approx(trajectory[-1][2]) == -(mass/alpha)*numpy.log( 1 + (10*alpha/mass)*dt)



def test_freefall_with_drag():


  simulation = Simulation()
  simulation.config.time_step = Q_(1,'s')
  simulation.config.magnus_coefficient *= 0
  simulation.config.drag_coefficient *= 10 # make drag really big

  alpha = simulation.config.drag_coefficient.to('kg/m').magnitude
  m = simulation.config.ball_mass.to('kg').magnitude
  g = simulation.config.gravitational_acceleration.to('m/s/s').magnitude
  v_0 = 0
  x_0 = 100

  def v(t):
    c = (numpy.sqrt(alpha / (m * g)) * v_0 - 1) / (numpy.sqrt(alpha / (m * g)) * v_0 + 1)
    v = numpy.sqrt(m*g/alpha) * (c*(numpy.exp(2*numpy.sqrt(g*alpha/m)*t))+1) / (1 - c* (numpy.exp(2*numpy.sqrt(g*alpha/m)*t)))
    return v
  def x(t):
    c = (numpy.sqrt(alpha / (m * g)) * v_0 - 1) / (numpy.sqrt(alpha / (m * g)) * v_0 + 1)
    x = x_0 - (m/(2*alpha))*numpy.log(abs((c*numpy.exp(2*numpy.sqrt(g*alpha/m) * t) - 1)**2 / ((numpy.exp(2*numpy.sqrt(g*alpha/m) * t)) * ((c-1)**2))))
    return x





  launch_config = LaunchConfiguration()
  launch_config.position = Q_(x_0,'m')*zhat
  trajectory = simulation.run(launch_config,lambda record: record[-1][3] < 1)


  dt = trajectory[-1][0]

  assert trajectory[-1][1] > -0.001
  assert trajectory[-1][1] < 0.001
  assert trajectory[-1][2] > -0.001
  assert trajectory[-1][2] < 0.001
  assert trajectory[-1][3] < 1

  assert trajectory[-1][4] > -0.001
  assert trajectory[-1][4] <  0.001
  assert trajectory[-1][5] > -0.001
  assert trajectory[-1][5] <  0.001
  assert trajectory[-1][6] <  0

  assert Approx(trajectory[-1][6]) == v(dt)
  assert Approx(trajectory[-1][3]).epsilon(0.25) == x(dt) # the analytic solution may not be quite rite here
  

def test_magnus_only():

  simulation = Simulation()
  simulation.config.time_step = Q_(1,'s')
  simulation.config.gravitational_acceleration*= 0
  simulation.config.drag_coefficient *= 0

  beta = simulation.config.magnus_coefficient.to_base_units().magnitude
  m = simulation.config.ball_mass.to('kg').magnitude
  omega = Q_(2000,'rpm').to('rad/s').magnitude
  v = 10

  # since the magnus force cannot change the kinetic energy of the ball,
  # we will get uniform circular motion. the radius is
  R = m / (beta * omega)
  print(R)


  # motion in the x-y plane
  # spin needs to be perpendicular to the velocity vector
  # in order to get uniform circular motion
  launch_config = LaunchConfiguration()
  launch_config.position = Q_(R,'m')*xhat
  launch_config.speed = Q_(v,'m/s')
  launch_config.direction = yhat
  launch_config.spin = Q_(omega,'rad/s')
  launch_config.spin_direction = zhat

  trajectory = simulation.run(launch_config,lambda record: record[-1][0] > 2)

  def x(t):
    return m*numpy.cos(v*beta*omega*t/m)/beta/omega
  def y(t):
    return m*numpy.sin(v*beta*omega*t/m)/beta/omega
  def vx(t):
    return -v*numpy.sin(v*beta*omega*t/m)
  def vy(t):
    return v*numpy.cos(v*beta*omega*t/m)

  dt = trajectory[-1][0]
  assert trajectory[-1][3] > -0.001
  assert trajectory[-1][3] <  0.001
  assert trajectory[-1][6] > -0.001
  assert trajectory[-1][6] <  0.001

  assert Approx(trajectory[-1][1]) == x(dt)
  assert Approx(trajectory[-1][2]) == y(dt)
  assert Approx(trajectory[-1][4]) == vx(dt)
  assert Approx(trajectory[-1][5]) == vy(dt)




  # motion in the x-z plane
  launch_config = LaunchConfiguration()
  launch_config.position = Q_(R,'m')*xhat
  launch_config.speed = Q_(v,'m/s')
  launch_config.direction = zhat
  launch_config.spin = Q_(omega,'rad/s')
  launch_config.spin_direction = -yhat

  trajectory = simulation.run(launch_config,lambda record: record[-1][0] > 2)

  def x(t):
    return m*numpy.cos(v*beta*omega*t/m)/beta/omega
  def z(t):
    return m*numpy.sin(v*beta*omega*t/m)/beta/omega
  def vx(t):
    return -v*numpy.sin(v*beta*omega*t/m)
  def vz(t):
    return v*numpy.cos(v*beta*omega*t/m)

  dt = trajectory[-1][0]
  assert trajectory[-1][2] > -0.001
  assert trajectory[-1][2] <  0.001
  assert trajectory[-1][5] > -0.001
  assert trajectory[-1][5] <  0.001

  assert Approx(trajectory[-1][1]) == x(dt)
  assert Approx(trajectory[-1][3]) == z(dt)
  assert Approx(trajectory[-1][4]) == vx(dt)
  assert Approx(trajectory[-1][6]) == vz(dt)



  # motion in the y-z plane
  launch_config = LaunchConfiguration()
  launch_config.position = Q_(R,'m')*yhat
  launch_config.speed = Q_(v,'m/s')
  launch_config.direction = zhat
  launch_config.spin = Q_(omega,'rad/s')
  launch_config.spin_direction = xhat

  trajectory = simulation.run(launch_config,lambda record: record[-1][0] > 2)

  def y(t):
    return m*numpy.cos(v*beta*omega*t/m)/beta/omega
  def z(t):
    return m*numpy.sin(v*beta*omega*t/m)/beta/omega
  def vy(t):
    return -v*numpy.sin(v*beta*omega*t/m)
  def vz(t):
    return v*numpy.cos(v*beta*omega*t/m)

  dt = trajectory[-1][0]
  assert trajectory[-1][1] > -0.001
  assert trajectory[-1][1] <  0.001
  assert trajectory[-1][4] > -0.001
  assert trajectory[-1][4] <  0.001

  assert Approx(trajectory[-1][2]) == y(dt)
  assert Approx(trajectory[-1][3]) == z(dt)
  assert Approx(trajectory[-1][5]) == vy(dt)
  assert Approx(trajectory[-1][6]) == vz(dt)



def test_magnus_only_linear_model():

  # test only magnus force with the linear velocity dependence.
  simulation = Simulation()
  simulation.config.time_step = Q_(1,'s')
  simulation.config.gravitational_acceleration*= 0
  simulation.config.drag_coefficient *= 0
  simulation.config.magnus_model = 'linear velocity'

  beta = simulation.config.magnus_coefficient.to_base_units().magnitude
  m = simulation.config.ball_mass.to('kg').magnitude
  omega = Q_(2000,'rpm').to('rad/s').magnitude
  v = 10

  # since the magnus force cannot change the kinetic energy of the ball,
  # we will get uniform circular motion. the radius is
  R = m * v / (beta * omega)


  # spin needs to be perpendicular to the velocity vector
  # in order to get uniform circular motion
  launch_config = LaunchConfiguration()
  launch_config.position = Q_(R,'m')*xhat
  launch_config.speed = Q_(v,'m/s')
  launch_config.direction = yhat
  launch_config.spin = Q_(omega,'rad/s')
  launch_config.spin_direction = zhat

  trajectory = simulation.run(launch_config,lambda record: record[-1][0] > 2)

  def x(t):
    return m*v*numpy.cos(beta*omega*t/m)/beta/omega
  def y(t):
    return m*v*numpy.sin(beta*omega*t/m)/beta/omega
  def vx(t):
    return -v*numpy.sin(beta*omega*t/m)
  def vy(t):
    return v*numpy.cos(beta*omega*t/m)

  dt = trajectory[-1][0]
  assert trajectory[-1][3] > -0.001
  assert trajectory[-1][3] <  0.001
  assert trajectory[-1][6] > -0.001
  assert trajectory[-1][6] <  0.001

  assert Approx(trajectory[-1][1]) == x(dt)
  assert Approx(trajectory[-1][2]) == y(dt)
  assert Approx(trajectory[-1][4]) == vx(dt)
  assert Approx(trajectory[-1][5]) == vy(dt)






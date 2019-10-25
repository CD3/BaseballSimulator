import yaml
import copy
import pint
import numpy

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity




xhat = numpy.array( [1,0,0] )
yhat = numpy.array( [0,1,0] )
zhat = numpy.array( [0,0,1] )

stride = Q_(5, 'ft')
shoulder_offset = Q_(-8,'inch')
shoulder_hieght = Q_(4,'foot') + Q_(8,'inch')
arm_length = Q_(30,'inch')

shoulder_position = Q_(60.5,'ft')*yhat - stride*yhat
shoulder_position += shoulder_offset*xhat
shoulder_position += shoulder_hieght*zhat



configs = {"configurations" : []}
for arm_angle in numpy.arange(0,91,5):
  theta = Q_(arm_angle,'degree')

  dx = -arm_length*numpy.sin(theta)
  dz = arm_length*numpy.cos(theta)

  pos = shoulder_position + dx*xhat + dz*zhat

  config = {
      'simulation' :{
        'time step' : '1 ms'
        },
      'launch' : {
        'position' : [str(q) for q in pos],
        'speed' : '95 mph',
        'direction' : [1,-55,1],
        'spin' : '2200 rpm',
        'spin_direction' : [float(numpy.cos(theta)),0,-float(numpy.sin(theta))],
        }
      }

  configs["configurations"].append(config)

with open("arm-angle-study.yaml", 'w') as f:
  yaml.dump(configs,f)


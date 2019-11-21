from .Simulator import *

import timeit
import hashlib
import pickle
import pathlib

import torch
import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
import plotly.graph_objs as go



class AimModels:

  class Base(torch.nn.Module):
    def __init__(self,pitch_types=[1], inputs=['effort','verticle_location','horizontal_location'], order=1, outputs=['verticle_deflection','horizontal_deflection']):
      super().__init__()
      self.pitch_types = pitch_types
      self.inputs = inputs
      self.order = order
      self.outputs = outputs

      self.in_features = len(pitch_types) + len(self.inputs)*self.order
      self.out_features = len(self.outputs)

    def save(self,filename):
      torch.save(self.state_dict(),filename)

    def load(self,filename):
      self.load_state_dict(torch.load(filename))

    def make_feature_vector(self,**kwargs):
      for input in ['pitch_type'] + self.inputs:
        if input not in kwargs:
          raise Exception(f"Error: required input '{input}' was not passed to make_feature_vector(...).")

      feature_vector = torch.zeros(self.in_features,dtype = ScalarType)
      # one-hot encode pitch type
      feature_vector[ self.pitch_types.index(kwargs['pitch_type']) ] = 1
      for n in range(self.order):
        n = n+1
        for i in range(len(self.inputs)):
          input = self.inputs[i]
          feature_vector[len(self.pitch_types)+(n-1)*len(self.inputs) + i] = kwargs[input]**n

      return feature_vector

    def make_output_vector(self,**kwargs):
      for output in self.outputs:
        if output not in kwargs:
          raise Exception(f"Error: required output '{output}' was not passed to make_output_vector(...).")

      output_vector = torch.zeros(self.out_features,dtype = ScalarType)
      for i in range(len(self.outputs)):
        output = self.outputs[i]
        output_vector[i] = kwargs[output]

      return output_vector






  class SimpleLinear(Base):
    def __init__(self,**kwargs):
      super().__init__(**kwargs)
      self.neuron = torch.nn.Linear( in_features=self.in_features, out_features=self.out_features )

    def forward(self,x):
      return self.neuron(x)

  class TwoLayerNetwork(Base):
    def __init__(self,**kwargs):
      super().__init__(**kwargs)
      self.L1 = torch.nn.Linear( in_features=self.in_features, out_features=N )
      self.L2 = torch.nn.Linear( in_features=N, out_features=self.out_features )

    def forward(self,x):
      return self.L2(torch.nn.functional.relu(self.L1(x)))


class Pitcher:
  def __init__(self):
    self.characteristics = {
        'release position': None,
        'pitches': None }
    self.aim_model = None


  def id(self):
    traits = { 'characteristics' : str(self.characteristics), 'aim_model' : str(self.aim_model) }
    return hashlib.sha1(str(traits).encode('utf8')).hexdigest()



  def configure_throw_from_deflection( self, type, effort, verticle_deflection, horizontal_deflection ):
    '''
    Create and return a LaunchConfiguration that corresponds to a throw.

    '''
    launch_config = LaunchConfiguration()
    launch_config.position = self.characteristics['release position']
    pitch = self.characteristics['pitches'][type]
    # allow specific pitches to override release point
    if 'release position' in pitch:
      launch_config.position = pitch['release position']
    
    launch_config.speed = pitch['velocity']*effort.to("").magnitude
    launch_config.spin = pitch['spin']
    launch_config.spin_direction = pitch['spin direction']
    launch_config.direction = -yhat


    # now diffect velocity and spin directions
    Rv = Rx(-verticle_deflection) # we want positive deflection to correspond to pointing "up"
    Rh = Rz(horizontal_deflection) # we want positive deflection to correspond to pointing "left" from pitchers point of view
    launch_config.deflect_direction( Rv )
    launch_config.deflect_direction( Rh )
    launch_config.deflect_spin_direction( Rv )
    launch_config.deflect_spin_direction( Rh )

    return launch_config

  def configure_throw( self, type, effort, verticle_location, horizontal_location):
    inputs = self.aim_model.make_feature_vector(
        pitch_type = type,
        effort = effort.to("percent").magnitude,
        verticle_location = verticle_location.to("m").magnitude,
        horizontal_location = horizontal_location.to("m").magnitude
        )

    outputs = self.aim_model(inputs)

    return self.configure_throw_from_deflection( type, effort, Q_(outputs[0],'degree'), Q_(outputs[1],'degree') )



  def train(self,simulation, epochs=100, num_throws = 1000, training_file = None, learning_rate=1e-4):
    '''
    Train the pitcher's aim model for a given simulation.
    '''
    # generate training data.
    # simulation inputs:
    #  - pitch type
    #  - effort
    #  - verticle deflection
    #  - horizontal deflection
    # simulation outputs:
    #  - verticle location
    #  - horizontal location

    if self.aim_model is None:
      raise Exception("Error: Pitcher's aim model has not been initialized. Cannot train.")

    N = num_throws
    if training_file is None:
      training_file = f'pitcher-training-data-{num_throws}-{self.id()}.pl'
    if isinstance(training_file,str):
      training_file = pathlib.Path(training_file).resolve()


    if training_file.is_file():
      print(f"Traning data file found ({str(training_file)}). Loading training data from file.")
      data = torch.load(str(training_file))
      sim_inputs = data['i']
      sim_outputs = data['o']
    else:
      print(f"No training data file found ({str(training_file)}). Created training dataset now")

      pint.set_application_registry(ureg)
      pool = Pool()


      sim_inputs  = dict()
      sim_inputs['type'] = torch.empty([N],dtype=int)
      sim_inputs['effort'] = torch.empty([N],dtype=ScalarType)
      sim_inputs['verticle_deflection'] = torch.empty([N],dtype=ScalarType)
      sim_inputs['horizontal_deflection'] = torch.empty([N],dtype=ScalarType)

      sim_outputs = dict()
      sim_outputs['verticle_location'] = torch.empty([N],dtype=ScalarType)
      sim_outputs['horizontal_location'] = torch.empty([N],dtype=ScalarType)

      configs = list()
      print("Generating training configurations.")
      for i in tqdm.tqdm(range(N)):
        type = numpy.random.choice(list(self.characteristics['pitches'].keys()))
        effort = numpy.random.uniform(75,105)
        verticle_deflection = numpy.random.normal(loc=0,scale=5)
        horizontal_deflection = numpy.random.normal(loc=0,scale=5)
        sim_inputs['type'][i] = int(type)
        sim_inputs['effort'][i] = float(effort)
        sim_inputs['verticle_deflection'][i] = float(verticle_deflection)
        sim_inputs['horizontal_deflection'][i] = float(horizontal_deflection)

        configs.append(self.configure_throw_from_deflection( type, Q_(effort,'percent'), Q_(verticle_deflection,'degree'), Q_(horizontal_deflection,'degree')))
     
      def terminate(record):
        if record[-1][2] < 0:
          return True
        if record[-1][0] > 2:
          return True
        return False

      def run_config(config):
        return simulation.run( config, terminate, record_all=False )

      # estimate the runtime
      # print("Estimating runtime to generate training data...")
      # i = 0
      # def run():
        # nonlocal i
        # run_config(configs[i%N])
        # i = i+1
      # runtime = timeit.timeit(run,number=10)/10
      # print(f"  will take approximately {runtime*N/pool.nodes} s to run {N} simulations @ {runtime} s / run on {pool.nodes} CPUs.")

      runs = list(tqdm.tqdm(pool.imap( run_config, configs), total=N))

      for i in range(N):
        sim_outputs['horizontal_location'][i] = runs[i][0][1]
        sim_outputs['verticle_location'][i] = runs[i][0][3]

      torch.save( {'i':sim_inputs,'o':sim_outputs}, str(training_file) )
      

    model_inputs  = torch.empty( [N,self.aim_model.in_features],dtype=ScalarType )
    model_outputs = torch.empty( [N,self.aim_model.out_features],dtype=ScalarType )

    for i in range(N):
      model_inputs[i,:] = self.aim_model.make_feature_vector( pitch_type=sim_inputs['type'][i],
                                                              effort=sim_inputs['effort'][i],
                                                              verticle_location=sim_outputs['verticle_location'][i],
                                                              horizontal_location=sim_outputs['horizontal_location'][i] )
    

      model_outputs[i,:] = self.aim_model.make_output_vector( verticle_deflection=sim_inputs['verticle_deflection'][i],
                                               horizontal_deflection=sim_inputs['horizontal_deflection'][i] )

    
    # optimizer = torch.optim.Adam( self.aim_model.parameters(), lr=1e-2 )
    optimizer = torch.optim.SGD( self.aim_model.parameters(), lr=learning_rate )
    loss_func = torch.nn.MSELoss()


    losses = list()
    print(f"Traning model:")
    for i in tqdm.tqdm(range(epochs)):
      optimizer.zero_grad()
      pred = self.aim_model(model_inputs)
      loss = loss_func(pred,model_outputs)
      losses.append(float(loss))
      loss.backward()
      optimizer.step()


    return losses



    



adam = Pitcher()
adam.characteristics['release position'] = Q_(-2,'ft')*xhat + Q_(55,'ft')*yhat + Q_(5.5,'ft')*zhat
adam.characteristics['pitches'] = { 1 : { 'velocity' : Q_(98,'mph'),
                   'spin' : Q_(2500,'rpm'),
                   'spin direction' : torch.tensor([-2,0.1,-1],dtype=ScalarType) }
}
adam.aim_model = AimModels.SimpleLinear(pitch_types=list(adam.characteristics['pitches'].keys()),order=1)



brian = Pitcher()
brian.characteristics['release position'] = Q_(-2,'ft')*xhat + Q_(55,'ft')*yhat + Q_(5.5,'ft')*zhat
brian.characteristics['pitches'] = { 
    1 : { 'velocity' : Q_(96,'mph'),
                   'spin' : Q_(2400,'rpm'),
                   'spin direction' : torch.tensor([-2,0.1,-1],dtype=ScalarType) },
    2 : { 'velocity' : Q_(82,'mph'),
                   'spin' : Q_(2200,'rpm'),
                   'spin direction' : torch.tensor([1,-0.1,0.2],dtype=ScalarType) }
}
brian.aim_model = AimModels.SimpleLinear(pitch_types=list(brian.characteristics['pitches'].keys()),order=1)




from .Simulator import *
from .Plotter import *
from . import Pitchers

from pathos.multiprocessing import ProcessingPool as Pool
import yaml
import sys
import pprint
import importlib
import copy
import tqdm

this_script = pathlib.Path(__file__).resolve()

# import ray
# ray.init()

from argparse import ArgumentParser


def main(argv):

  parser = ArgumentParser(description="A baseball pitch simulator.")

  parser.add_argument("config_file",
                      action="store",
                      help="Configuration file." )
  parser.add_argument("-l", "--list-pitchers",
                      action="store_true",
                      help="List the name of all built-in pitcher available for use." )
  parser.add_argument("-p", "--show-performance",
                      action="store_true",
                      help="Generate plot(s) showing model performance after training." )



  args = parser.parse_args(argv)

  if args.list_pitchers:
    pitchers = [ p for p in dir(Pitchers) if isinstance( getattr(Pitchers,p), Pitchers.Pitcher ) ]
    print("Available pitchers:")
    for p in pitchers:
      print("\t",p)
    sys.exit(0)

  # load config
  config_file = pathlib.Path(args.config_file).resolve()
  with config_file.open() as f:
    conf = yaml.safe_load(f)

  pitcher = None
  pitcher_name = conf.get("pitcher",dict()).get("name",None)
  if pitcher_name is not None:
    pitcher = copy.deepcopy(getattr(Pitchers,pitcher_name))

  # setup data files
  training_conf = conf.get("training",dict())
  training_data_file_template = training_conf.get("training data file","{name}-training-data-{num}-{id}.yaml")
  input_model_file_template = training_conf.get("input model file","{name}-model-{id}.yaml")
  output_model_file_template = training_conf.get("output model file","{name}-model-{id}-new.yaml")
  num_training_trials = training_conf.get("number trials",1000)
  num_training_epochs = training_conf.get("number epochs",100)


  context = dict()
  context["name"] = pitcher_name
  context["id"] = pitcher.id()
  context["num"] = num_training_trials

  training_data_file = pathlib.Path(training_data_file_template.format(**context)).resolve()
  input_model_file = pathlib.Path(input_model_file_template.format(**context)).resolve()
  output_model_file = pathlib.Path(output_model_file_template.format(**context)).resolve()

  sim = Simulation()
  sim.configure(conf.get('simulation',dict()))

  if input_model_file.is_file():
    pitcher.aim_model.load(str(input_model_file))
  else:
    print(f"WARNING: '{str(input_model_file)}' does not exist.")
  losses = pitcher.train(sim,num_training_epochs,num_training_trials,training_data_file)

  if output_model_file.is_file():
    print(f"WARNING: '{str(output_model_file)}' already exists. It will be OVERWRITTEN.")
  pitcher.aim_model.save(str(output_model_file))

  print("Summary:")
  print(f"\tinitial training loss: {losses[0]}")
  print(f"\t  final training loss: {losses[-1]}")


  if args.show_performance:
    print("Evaluating pitcher")
    configs = list()

    aim_locations = list()
    for x in numpy.arange( -15, 15+1,5 ):
      for z in numpy.arange( 12,5*12+1, 12 ):
        aim_x = Q_(x,'inch')
        aim_z = Q_(z,'inch')
        config = pitcher.configure_throw( 1, Q_(100,'percent'), aim_z, aim_x)
        aim_locations.append( [aim_x,aim_z] )
        configs.append(config)

    def compute_location(config):
      trajectory = sim.run(config, terminate_function=lambda x: x[-1][2] < 0, record_all=False)
      act_x = Q_(trajectory[0][1],'m')
      act_z = Q_(trajectory[0][3],'m')
      return [act_x,act_z]

    pool = Pool()
    locations = list(tqdm.tqdm(pool.imap( compute_location, configs), total=len(configs)))

    txs = [ r[0].to("in").magnitude for r in aim_locations ]
    tzs = [ r[1].to("in").magnitude for r in aim_locations ]
    axs = [ r[0].to("in").magnitude for r in locations ]
    azs = [ r[1].to("in").magnitude for r in locations ]


    fig = go.Figure(data=[go.Scatter(x=txs,y=tzs,mode='markers'),go.Scatter(x=axs,y=azs,mode='markers')])
    fig.show()









if __name__ == "__main__":
  main(sys.argv[1:])


from BaseballSimulator.Simulator import *
from BaseballSimulator.Plotter import *
from BaseballSimulator import Pitchers
from BaseballSimulator.pdict import *

from pathos.multiprocessing import ProcessingPool as Pool
import yaml
import sys
import pprint
import importlib
import copy

import click

this_script = pathlib.Path(__file__).resolve()

# import ray
# ray.init()

from argparse import ArgumentParser


pitchers_cache = pdict()


def construct_pitcher(config):
  '''
  Constructs and returns an instance of Pitcher given a configuration. Each unique pitcher configuration is cached,
  so multiple instances of the same configuration will use the same object.
  '''
  config = pdict(config)

  pitcher = None
  name = config.get('name',None)
  aim_model = config.get('aim_model',None)
  if name is not None and aim_model is not None:
    if name in pitchers_cache and aim_model in pitchers_cache[name]:
      # pitcher has already been created
      pitcher = pitchers_cache[name][aim_model]
    else:
      # pitcher has not already been created
      # either the pitcher with given name has not been created,
      # or a pitcher with the same name but a different aim model has been created.
      # in either case, we want a new copy of the pitcher with the given name.
      pitcher = copy.deepcopy(getattr( Pitchers, name ))

      # add name to cache if it is new
      if name not in pitchers_cache:
        pitchers_cache[name] = dict()

      # set aim model
      file = pathlib.Path(aim_model)
      if not file.is_file():
        raise Exception(f"Could not find '{file}'.")
      pitcher.aim_model.load( str(file) )

      # add pitcher to cache
      pitchers_cache[name][aim_model] = pitcher

  return pitcher

@click.command("A baseball pitch simulator.")
@click.argument("config_file")
@click.option("--list-pitchers","-l",is_flag=True,help="List the name of all built-in pitcher available for use.")
@click.option("--output-config-file","-o",help="Name of file to write launch-sim.py configuration file to.")
@click.option("--run-launch-sim","-r",help="Run launch-sim.py after generating configuration.")
def main(config_file,list_pitchers,output_config_file,run_launch_sim):

  if args.list_pitchers:
    pitchers = [ p for p in dir(Pitchers) if isinstance( getattr(Pitchers,p), Pitchers.Pitcher ) ]
    print("Available pitchers:")
    for p in pitchers:
      print("\t",p)
    sys.exit(0)

  # load pitch-sim config
  config_file = pathlib.Path(args.config_file).resolve()
  with config_file.open() as f:
    pconf = yaml.safe_load(f)

  # generate launch-sim config
  lconf = {'configurations':[]}
  for conf in pconf.get('configurations',[]):
    conf = pdict(conf)
    c = dict()
    c['simulation'] = conf.get('simulation',pdict()).dict()
    c['launch'] = dict()

    # get or create pitcher
    pitcher = construct_pitcher(conf['pitcher'])

    
    type = conf['pitch']['type']
    effort = conf['pitch']['effort']
    location_x = conf['pitch']['location'][0]
    location_z = conf['pitch']['location'][1]
    sconf = pitcher.configure_throw(type,Q_(effort),Q_(location_z),Q_(location_x))

    c['launch']['position'] = [str(x) for x in sconf.position]
    c['launch']['speed'] = str(sconf.speed)
    c['launch']['direction'] = [float(x) for x in sconf.direction]
    c['launch']['spin'] = str(sconf.spin)
    c['launch']['spin_direction'] = [float(x) for x in sconf.spin_direction]

    lconf['configurations'].append(c)



  output_config_file = pathlib.Path(args.output_config_file).resolve()
  output_config_file.write_text(yaml.dump(lconf))

  if args.run_launch_sim:
    launch_sim = importlib.import_module(".launch-sim",__package__)
    launch_sim.main([str(output_config_file),"-p"])



if __name__ == "__main__":
  main()


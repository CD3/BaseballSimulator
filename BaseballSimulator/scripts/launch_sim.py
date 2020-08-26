from BaseballSimulator.Simulator import *
from BaseballSimulator.Plotter import *
from BaseballSimulator.pdict import *

from pathos.multiprocessing import ProcessingPool as Pool
import yaml
import sys
import importlib

import click

@click.command(help="A projectile motion simulator.")
@click.argument("config_file")
@click.option("--write-to-file","-w",is_flag=True,help="Write trajectories to file.")
@click.option("--output-file","-o",help="Output file basename. Multiple simulations will be written to seprate files with index appended.")
@click.option("--output-format","-f",default="txt",help="Output file format.")
@click.option("--display-plots","-p",is_flag=True,help="Generate and display plot of trajectories.")
@click.option("--terminate","-t",help="Specify python file containing 'terminate' function to be used.")
@click.option("--serial","-s",is_flag=True,help="Run simulations in series rather than parallel.")
def main(config_file,write_to_file,output_file,output_format,display_plots,terminate,serial):

  def load_configs_from_file(filename):
    configs = []
    with open(filename,'r') as f:
      fconf = yaml.safe_load(f)

    for config in fconf.get('configurations',[]):
      c = [Simulation(),LaunchConfiguration()]
      c[0].configure( config.get('simulation',{}) )
      c[1].configure( config.get('launch',{}) )
      configs.append(c)

    return configs

  configs = load_configs_from_file(config_file)

  # @ray.remote
  def run_configuration(sim_and_launch):
    try:
      if terminate:
        file = pathlib.Path(terminate).resolve()
        path = file.parent
        name = file.stem
        term = importlib.import_module(name)
        terminate = term.terminate
    except:
      def terminate(record):
        # if the ball is below ground, bounce it
        if record[-1][3] < 0:
          record[-1][3] *= -1
          record[-1][6] *= -1

        # terminate when ball reaches back of plate
        if record[-1][2] < 0:
          return True

        # make sure we don't get stuck in an infinite loop
        if record[-1][0] > 10:
          return True
        

        return False

    return sim_and_launch[0].run( sim_and_launch[1], terminate )

  if serial:
    trajectories = [ run_configuration(c) for c in configs ]
  else:
    processes = Pool()
    trajectories = processes.map(run_configuration, configs)
    # trajectories = ray.get( [run_configuration.remote(c) for c in configs] )


  if write_to_file:
    for i,trajectory in enumerate(trajectories):
      if output_format == 'txt':
        with open(f"{output_file}-{i}.txt", 'w') as f:
          f.write("#t x y z vx vy vz wx wy wz")
          f.write("\n")
          for state in trajectory:
            line = " ".join( [str(elem.item()) for elem in state] )
            f.write(line)
            f.write("\n")
      elif output_format == 'pt':
        torch.save( torch.stack(trajectory), f"{output_file}-{i}.pt")

      else:
        raise Exception(f"Error: Unrecognized output file format ({output_format})")

  if display_plots:
    plotter = Trajectory3DPlot()
    # add stikezone graphic
    W = Q_(17,'inch').to("m")
    H = Q_(2.5,'ft').to("m")
    b = Q_(1.5,'ft').to("m")
    home_plate_corners = list()
    home_plate_corners.append( 0*yhat )
    home_plate_corners.append( W/2*yhat + W/2*xhat )
    home_plate_corners.append( W*yhat + W/2*xhat )
    home_plate_corners.append( W*yhat - W/2*xhat )
    home_plate_corners.append( W/2*yhat - W/2*xhat )
    home_plate_corners.append( 0*yhat )
    home_plate_plot = go.Scatter3d(
        x = [ r[0] for r in home_plate_corners ],
        y = [ r[1] for r in home_plate_corners ],
        z = [ r[2] for r in home_plate_corners ],
        mode='lines',
        line=dict(color='rgb(255, 0, 0)'),
        name='Home Plate'
    ) 
    stike_zone_corners = list()
    stike_zone_corners.append( W*yhat + W/2*xhat + b*zhat )
    stike_zone_corners.append( W*yhat + W/2*xhat + (b+H)*zhat )
    stike_zone_corners.append( W*yhat - W/2*xhat + (b+H)*zhat )
    stike_zone_corners.append( W*yhat - W/2*xhat + b*zhat )
    stike_zone_corners.append( W*yhat + W/2*xhat + b*zhat )
    stike_zone_plot = go.Scatter3d(
        x = [ r[0] for r in stike_zone_corners ],
        y = [ r[1] for r in stike_zone_corners ],
        z = [ r[2] for r in stike_zone_corners ],
        mode='lines',
        line=dict(color='rgb(255, 0, 0)'),
        name='Strike Zone'
    ) 
    plotter.plot( trajectories, [home_plate_plot, stike_zone_plot] )


if __name__ == "__main__":
  main()

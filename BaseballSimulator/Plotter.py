import plotly
import plotly.graph_objs as go
import torch
import yaml
import pathlib



class Trajectory3DPlot:
  def __init__(self):
    plot_layout_yaml = '''
scene:
  xaxis:
    title: x
    range: [-1.5, 1.5]
  yaxis:
    title: y
    range: [0, 20]
  zaxis:
     title: z
     range: [0, 2.5]
  camera:
    up:
      x: 0
      y: 0
      z: 1
    center:
      x: 0
      y: 0
      z: 0
    eye:
      x: 0
      y: -2
      z: 0
  aspectratio:
    x: 0.5
    y: 3.5
    z: 0.5
  aspectmode: manual
    '''

    self.layout = yaml.safe_load(plot_layout_yaml)
    self.data_dir = pathlib.Path("Figures").resolve

  def plot(self,trajectories,data=list()):
    for trajectory in trajectories:
      trajectory = torch.stack(trajectory)
      data.append( go.Scatter3d({'x':trajectory[:,1],'y':trajectory[:,2],'z':trajectory[:,3],'mode':'lines'}) )
    fig = go.Figure(data=data, layout=self.layout)
    fig.show()
    # plotly.offline.plot(fig, filename='./Figures/' + name + '.html', auto_open=True)


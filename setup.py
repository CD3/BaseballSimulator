from setuptools import setup, find_packages

setup(
    name='BaseballSimulator',
    version = '0.1',
    packages=find_packages(),
    install_requires=[
        'pint',
        'torch',
        'plotly',
        'pyyaml',
        'numpy',
        'scipy',
        'pathos',
        'click',
        'tqdm',
    ],
    # entry_points='''
    # [console_scripts]
    #  launch-sim=BaseballSimulator.scripts.launch_sim:main
    #  pitch-sim=BaseballSimulator.scripts.pitch_sim:main
    #  pitch-trainer=BaseballSimulator.scripts.pitch_trainer:main
    # ''',
)

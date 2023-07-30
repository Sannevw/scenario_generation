# Supplementary Material

This supplementary material contains the code and video for the IROS'23 paper _"Generating Scenarios from High-Level Specifications for Object Rearrangement Tasks"_. 

__Abstract__

Rearranging objects is an essential skill for robots. To quickly teach robots new rearrangements tasks, we would like to generate training scenarios from high-level specifications that define the relative placement of objects for the task at hand. Ideally, to guide the robot's learning we also want to be able to rank these scenarios according to their difficulty. Prior work has shown how diverse scenario generation from specifications and providing the robot with easy-to-difficult samples can improve the learning. Yet, existing scenario generation methods typically cannot generate diverse scenarios while controlling their difficulty. We address this challenge by conditioning generative models on spatial logic specifications to generate spatially-structured scenarios that meet the specification and desired difficulty level. Our experiments showed that generative models are more effective and data-efficient than rejection sampling and that the spatially-structured scenarios can drastically improve training of downstream tasks by orders of magnitude.


## Video
The [Supplementary Video](video/IROS-SupplementaryVideo.mp4) shows generated scenarios for each of the three experiments.

## Code
The code contains three interactive Jupyter notebooks in which users can choose desired satisfaction values and generate/render scenarios. 
The code has been tested on macOS 12.3 with Python 3.8 in an Anaconda environment. 

## Installation

Running this code requires Python 3.8.
The experiments use [Python Poetry](https://python-poetry.org) for packages and dependency management. Please make sure you have Poetry installed.

To clone the repository and install the dependencies:
```shell
git clone https://github.com/sannevw/scenario_generation.git
poetry install
```

### Reproduce results from paper

The notebooks are named "IROS23-Experiment-X" where X indicates the experiment number. The notebooks automatically load our backend and the trained models. 
You should now be able to run the Jupyter Notebooks (opening in your web browser):
```shell
poetry run jupyter notebook
```

This repository contains the following notebooks:
- [IROS23-Experiment-1](IROS23-Experiment-1.ipynb)
- [IROS23-Experiment-2](IROS23-Experiment-2.ipynb)
- [IROS23-Experiment-3](IROS23-Experiment-3.ipynb)

## Repository Structure
- [files](./files): The data and model files used in the experiments
- [gigalib](./gigalib): Scripts to reproduce the experiments
- [urdf](./urdf): Files for rendering the results
- [video](./video): The supplementary video to the IROS'23 paper

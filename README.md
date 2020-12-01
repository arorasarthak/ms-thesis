# ms-thesis
Repo for Master's Thesis Work for "Perception Methods for Speed and Separation Monitoring"

### Abstract
This work presents the development of a perception pipeline to passively track the partial human ground pose in the context of human robot collaboration. The main motivation behind this work is to provide a speed and separation monitoring based safety controller with an estimate of human position on the factory floor. Three time-of-flight sensing rings affixed to the major links of an industrial manipulator are used to implement the aforementioned. Along with a convolutional neural network based unknown obstacle detection strategy, the ground position of the human operator is estimated and tracked using sparse 3-D point inputs. Experiments to analyze the viability of our approach are presented in depth in the further sections which involve real-world and synthetic datasets. Ultimately, it is shown that the sensing system can provide reliable information intermittently and can be used for higher level perception schemes.

### Requirements
- Tensorflow 2.0 or above
- Python 3.6 or above 
- CoppeliaSim (V-REP) 3.6 or above
- PyZMQ
- PyQtGraph
- PyRep
- Universal Robots - RTDE
- MATLAB Engine API for Python

### Summary
This repo is a code base for my master's thesis. The entire system was implemented in python using CoppeliaSim Robotics Simulator. The project also heavily depends on zero-mq for communication. All the python programs live in the `scripts` folder and there are two scripts that are mainly responsible for everything, namely: `scripts/experiment.py` and `scripts/rtde_helper.py`. The system pipeline originates from raw sensor data obtained from the simulation(Universal Robots UR10 and time-of-sensors) and gets passed through tensorflow models along with bayesian filter implementations using the MATLAB API for Python

### Links
[Manuscript](https://scholarworks.rit.edu/theses/10334/)

[![Video](https://img.youtube.com/vi/fxHwCIYJh8I/0.jpg)](https://www.youtube.com/watch?v=fxHwCIYJh8I "Video")

Click on the image to watch the video!

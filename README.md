# qn_optimizer

## Overview

The qn_optimizer package provides a library for Quasi-Newton Optimization. This algorithm can be used to minimize an arbitrary, user-defined objective function.

**Keywords:** optimization quasi-newton

### License

The source code is released under a [MIT license](LICENSE).

**Author: Paul D'Angio<br />
Maintainer: Paul D'Angio, pcdangio@gmail.com**

The qn_optimizer package has been tested under [ROS] Melodic and Ubuntu 18.04. This is research code, expect that it changes often and any fitness for a particular purpose is disclaimed.

## Installation

### Building from Source

#### Dependencies

- [Robot Operating System (ROS)](http://wiki.ros.org)
- [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page)

#### Building

To build from source, clone the latest version from this repository into your catkin workspace and compile the package using

        cd catkin_workspace/src
        git clone https://github.com/pcdangio/ros-qn_optimizer.git qn_optimizer
        cd ../
        catkin_make

## Usage

See the [examples](https://github.com/pcdangio/ros-qn_optimizer/tree/master/examples) for how to use this library.

### Objective Function

The library requires the user to specify an objective function that the optimizer will seek to minimize. This objective function may have any number of dimensions (specified in the constructor as n_dimensions), and may be linear or nonlinear.  It may also implement optimization constrains using barrier/penalty functions within.

### Gradient of Objective Function

The user may optionally specify a gradient function for the objective function.  If no gradient function is provided by the user, the library will use small perturbations of the objective function to estimate the gradient during optimization (at the cost of higher computational complexity).

## Bugs & Feature Requests

Please report bugs and request features using the [Issue Tracker](https://github.com/pcdangio/ros-qn_optimizer/issues).

[ROS]: http://www.ros.org
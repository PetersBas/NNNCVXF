# NNNCVXF - **N**eural **N**etwork training via **N**on-**C**on**V**e**x** **F**easibility
**N**eural **N**etwork training via **N**on-**C**on**V**e**x** **F**easibility

This repository contains scripts to reproduce the examples from:

```bibtex
@article{Peters2021OutputConstrainedNetworks,
  title={Point-to-set distance functions for output-constrained neural networks},
  author={Peters, Bas},
  year={2021}
}
``` 

This software is not intended as a general neural network toolbox. Some generalizations of the current software are planned in the near future.


## Installation for Julia 1.5:
 ```
 add https://github.com/PetersBas/NNNCVXF.git
 ``` 
 
  - NNNCVXF also depends on the packages: 
 	- [SetIntersectionProjection]( https://github.com/slimgroup/SetIntersectionProjection.jl)
 	- [PyPlot](https://github.com/JuliaPy/PyPlot.jl)
 	- [InvertibleNetworks](https://github.com/slimgroup/InvertibleNetworks.jl)
  	- [TestImages](https://github.com/JuliaImages/TestImages.jl) 
 	- [Images](https://github.com/JuliaImages/Images.jl) 
 	- [Flux](https://github.com/FluxML/Flux.jl)
	
## Examples:
 - [Time-Lapse HyperSpectral land-use segmentation](https://github.com/PetersBas/NNNCVXF/blob/main/examples/TimeLapseHyperspectralConstrained.jl) Segment two 3D data volumes of hyperspectral data to identify land use. Assumes just 20 point annotations for class one, no annotation for class 2, and prior knowledge on the percentage of surface area that experienced land-use change. (set up for GPU)
 - [CamVid street scenes](https://github.com/PetersBas/NNNCVXF/blob/main/examples/ConstrainedCamvid.jl) Segment 2D RGB images into different object types. In this modified CamVid experiment, we use just 47 images for training and 15 for validation, with 8 point annotations per class per image. We also assume approximate information on the anisotropic total-variation of the training images. (set up for GPU)
 - Single image segmentation with corruption [Zebra](https://github.com/PetersBas/NNNCVXF/blob/main/examples/zebra_stripes_minkowski.jl) [Bike](https://github.com/PetersBas/NNNCVXF/blob/main/examples/motorbike_stripes_minkowski.jl) These experiments train a neural network to segment a single image based on a bounding box and some prior knowledge. No pre-training was used. The network is trained from scratch for each image. Images contain coherent corruption in the form of missing rows. Prior knowledge consists of a 'simple' image description using a Minkowski set that is the sum of monotonically increasing and decreasing image components, as well as rough bounds on the size of the object. (set up for CPU, although GPU would be faster). See also [Bike (Grabcut,Python)](https://github.com/PetersBas/NNNCVXF/blob/main/examples/grabcut_bbox_bike.py) and [Zebra (Grabcut,Python)](https://github.com/PetersBas/NNNCVXF/blob/main/examples/grabcut_bbox_zebra.py)

## Required Data:
Data is provided for the single image segmentation with corruption examples [Zebra](https://github.com/PetersBas/NNNCVXF/blob/main/examples/zebra_stripes_minkowski.jl) [Bike](https://github.com/PetersBas/NNNCVXF/blob/main/examples/motorbike_stripes_minkowski.jl). Data files are larger for [Time-Lapse HyperSpectral land-use segmentation](https://github.com/PetersBas/NNNCVXF/blob/main/examples/TimeLapseHyperspectralConstrained.jl) and [CamVid street scenes](https://github.com/PetersBas/NNNCVXF/blob/main/examples/ConstrainedCamvid.jl). Download instructions are included in those two scripts.

##Basic Code Functionality:
All examples follow the same workflow
 1. Set up training (and possibly validation) data and labels (if any are available).
 2. Set up projection operators that project onto the intersection of constraint sets, implemented by [SetIntersectionProjection](https://github.com/slimgroup/SetIntersectionProjection.jl)
 3. Set up the neural network. This code fixes the network to be a [fully reversible (invertible) hyperbolic network](https://arxiv.org/abs/1905.10484), implemented by [InvertibleNetworks](https://github.com/slimgroup/InvertibleNetworks.jl). You can still set the length, kernel sizes, width, number of input and output channels.
 4. Train the network. This uses gradient descent (based) methods. The loss and gradient are computed via the squared distance of the neural network output to the intersection of constraint sets. This work shows that the gradient computation is possible via standard adjoint-state/backpropagation: 1) forward propagation of the input data through the network. 2) compute loss and final Lagrangian multiplier. 3) interleave backward propagation to obtain the other Lagrangian multipliers in reverse order, with the gradient computation for network parameters. 4) update network parameters at the end
 5. Predict and plot results. After training, we obtain a prediction (that does not use any constraints) by forward propagating the input data though the network. The output is then the predicted probability per class for each pixel. Plotting shows things like loss function per iteration, network output per class, thresholded network ouput showing the most likely class for each pixel, and data + predition plotted on top of each other.
 
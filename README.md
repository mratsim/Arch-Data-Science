# Data Science packages for Archlinux

Welcome to my repo to build Data Science, Machine Learning, Computer Vision, Natural language Processing and Deep Learning packages from source.

My aim is to squeeze the maximum performance for my current configuration (Skylake Xeon + Nvidia Pascal GPU) so:

* All packages are build with -O3 -march=native if the package ignores /etc/makepkg.conf config.
* I do not use fast-math except if it's the default upstream (example opencv). You might want to enable it for GCC and NVCC (Nvidia compiler), for example for Theano
* All CUDA packages are build with CUDA 8, cuDNN 5.1 and Compute capabilities 6.1 (Pascal)
* Pytorch is also build with MAGMA support. Magma is a linear algebra library for heterogeneous computing (CPU + GPU hybridization)
* BLAS library is OpenBLAS except for Tensorflow (Eigen)
* Parallel library is OpenMP except for Tensorflow (Eigen) and OpenCV (Intel TBB, Thread Building Blocks)  
* OpenCV is further optimized with Intel IPP (Integrated Performance Primitives)
* Nvidia cuBLAS and cuFFT are used wherever possible

My Data Science environment is running from a LXC container so Tensorflow build system, bazel, must be build with its auto-sandboxing disabled.

## Caveats
Please note that current mxnet and lightgbm packages are working but must be improved: they put their libraries in /usr/mxnet and /usr/lightgbm
Packages included are those not available by default in Archlinux AUR or that needed substantial modifications. So check Archlinux AUR for standard packages like Numpy, Pandas or Theano.

# Data Science packages for Archlinux

Welcome to my repo to build Data Science, Machine Learning, Computer Vision, Natural language Processing and Deep Learning packages from source.

## Performance considerations

My aim is to squeeze the maximum performance for my current configuration (Skylake-X i9-9980XE + 2x RTX 2080Ti) so:

* All packages are build with -O3 -march=native if the package ignores /etc/makepkg.conf config.
* I do not use fast-math except if it's the default upstream (example opencv). You might want to enable it for GCC and NVCC (Nvidia compiler)
* All CUDA packages are build with CUDA 10.1, cuDNN 7.6 and Compute capabilities 7.5 (Turing).
* Pytorch is build
  * with MAGMA support. Magma is a linear algebra library for heterogeneous computing (CPU + GPU hybridization)
  * with MKLDNN support. MKLDNN is a optimized x86 backend for deep learning.
* BLAS library is MKL except for Tensorflow (Eigen).
* Parallel library is Intel OpenMP except for Tensorflow (Eigen), PyTorch (because linking is buggy) and OpenCV (Intel TBB, because linking is buggy as well)
* OpenCV is further optimized with Intel IPP (Integrated Performance Primitives)
* Nvidia libraries (CuBLAS, CuFFT, CuSPARSE ...) are used wherever possible

If running in a LXC container, bazel (necessary to build Tensorflow), must be build with its auto-sandboxing disabled.

## Caveats
Please note that current mxnet and lightgbm packages are working but must be improved: they put their libraries in /usr/mxnet and /usr/lightgbm
Packages included are those not available by default in Archlinux AUR or that needed substantial modifications. So check Archlinux AUR for standard packages like Numpy or Pandas.

## Suggestions

Beyond the packages provided here, here are some useful tools:
* CSV manipulation from command-line
    * [xsv](https://github.com/BurntSushi/xsv) - The fastest, multi-processing CSV library. Written in Rust.
* Geographical data (combined them with a clustering algorithm)
    * Geopy
    * Shapely
* GPU computation
    * Nvidia's RAPIDS (to be wrapped)
      * [GPU Dataframes](https://github.com/rapidsai/cudf)
      * [Sklearn-like on GPU](https://github.com/rapidsai/cuml)
* Monitoring
    * htop - Monitor CPU, RAM, load, kill programs
    * [nvtop](https://github.com/Syllo/nvtop) - Monitor Nvidia GPU
    * nvidia-smi - Monitor Nvidia GPU (included with nvidia driver)
        1. nvidia-smi -q -g 0 -d TEMPERATURE,POWER,CLOCK,MEMORY -l #Flags can be UTILIZATION, PERFORMANCE (on Tesla) ...
        2. nvidia-smi dmon
        3. nvidia-smi -l 1
* Rapid prototyping, Research
    * Jupyter - Code Python, R, Haskell, Julia with direct feedback in your browser
    * jupyter_contrib_nbextensions - Extensions for jupyter (commenting code, ...)
* Text
    * gensim - word2vec
* Time data
    * Workalendar - Business calendar for multiple countries
* Video
    * Vapoursynth - Frameserver for video pre-processing
* Visualization
    * The [Vega ecosystem](https://vega.github.io/)
      * [Altair](https://github.com/altair-viz/altair) - declarative data visualization
      * [Voyager](https://github.com/vega/voyager) - Automatic Exploratory Data Analysis
      * [Lyra](https://github.com/vega/lyra) - Tableau-like data visualization design
    * [Seaborn](https://github.com/mwaskom/seaborn)
    * [Plot.ly](https://github.com/plotly/plotly.py)

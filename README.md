# Data Science packages for Archlinux

Welcome to my repo to build Data Science, Machine Learning, Computer Vision, Natural language Processing and Deep Learning packages from source.

## Performance considerations

My aim is to squeeze the maximum performance for my current configuration (Skylake Xeon + Nvidia Pascal GPU) so:

* All packages are build with -O3 -march=native if the package ignores /etc/makepkg.conf config.
* I do not use fast-math except if it's the default upstream (example opencv). You might want to enable it for GCC and NVCC (Nvidia compiler), for example for Theano
* All CUDA packages are build with CUDA 8, cuDNN 5.1 and Compute capabilities 6.1 (Pascal)
* Pytorch is also build with MAGMA support. Magma is a linear algebra library for heterogeneous computing (CPU + GPU hybridization)
* BLAS library is OpenBLAS except for Tensorflow (Eigen)
* Parallel library is OpenMP except for Tensorflow (Eigen) and OpenCV (Intel TBB, Thread Building Blocks)  
* OpenCV is further optimized with Intel IPP (Integrated Performance Primitives)
* Nvidia libraries (CuBLAS, CuFFT, CuSPARSE ...) are used wherever possible

My Data Science environment is running from a LXC container so Tensorflow build system, bazel, must be build with its auto-sandboxing disabled.

## Caveats
Please note that current mxnet and lightgbm packages are working but must be improved: they put their libraries in /usr/mxnet and /usr/lightgbm
Packages included are those not available by default in Archlinux AUR or that needed substantial modifications. So check Archlinux AUR for standard packages like Numpy, Pandas or Theano.

## Description of the Data Science Stack
Packages not described here are dependencies of others (bazel -> Tensorflow, murmurhash, plac, preshed, etc -> spaCy)

* General packages
    * Monitoring
        * htop - Monitor CPU, RAM, load, kill programs
        * nvidia-smi - Monitor Nvidia GPU
            1. nvidia-smi -q -g 0 -d TEMPERATURE,POWER,CLOCK,MEMORY -l #Flags can be UTILIZATION, PERFORMANCE (on Tesla) ...
            2. nvidia-smi dmon
            3. nvidia-smi -l 1
    * CSV manipulation from command-line
        * [xsv](https://github.com/BurntSushi/xsv) - The fastest, multi-processing CSV library. Written in Rust.
    * Computation, Matrix, Scientific libraries
        * OpenBLAS + LAPACK - Efficient Matrix computation and Linear Algebra library (alternative MKL)
        * Numpy - Matrix Manipulation in Python
        * Scipy - General scientific library for Python. Sparse matrices support
    * Rapid Development, Research
        * Jupyter - Code Python, R, Haskell, Julia with direct feedback in your browser
        * jupyter_contrib_nbextensions - Extensions for jupyter (commenting code, ...)
    * GPU computation
        * CUDA - Nvidia API for GPGPU
        * CUDNN - Nvidia primitives for Neural Networks
        * Magma - Linear Algebra for OpenCL and CUDA and heteregenous many-core systems
    * Visualization, Exploratory Data Analysis
        * Matplotlib
        * Seaborn

* Machine Learning
    * Data manipulation
        * Pandas - Dataframe library
        * Dask - Dataframe library for out-of-core processing (Data that doesn't fit in RAM)
        * Scikit-learn-pandas - Use Pandas Dataframe seemlessly in Scikit-learn pipelines
        * Numexpr
    * Multicore processing
        * joblib
        * Numba
        * concurrent.futures
        * Dask
        * paratext - fast CSV to pandas
    * Compressing, storing data
        * Bcolz - Compress Numpy arrays in memory or on-disk and use them transparently
        * Zarr - Compress Numpy array in memory or on-disk and use them transparently
    * Out-of-core processing
        * Bcolz, Zarr
        * Dask
    * Structured Data - Classifier
        * Scikit-learn - General ML framework
        * XGBoost - Gradient Boosted tree library
        * LightGBM - Gradient Boosted tree library
        _XGBoost and LightGBM classifiers should be preferred to Scikit-learn_
    * Pipelines
        * Scikit-learn

        _I don't recommend Scikit-learn pipelines as they are not flexible enough: not possible to use a validation set for XGBoost/LightGBM for early-stopping, computation waste for features that don't use target labels._
    * Unsupervised Learning - High cardinality/dimensionality (PCA, SVD, ...)
        * Scikit-learn

        _Scikit-learn manifold implementations like t-SNE are not recommended for efficiency (RAM, Computation) reasons_
    * Geographical data, Clustering
        * scikit-learn
        * Geopy
        * Shapely
        * HDBSCAN - Density-based clustering
    * Categorical data
        * python-categorical-encoders - Encoding with One-Hot, Binary, N-ary, Feature hashes and other scheme.

        _Scikit-learn One-Hot Encoding, LabelEncoding, LabelBinarizer are a mess API-wise but are efficient if wrapped properly_
    * Stacking
        * mlxtend

        _I recommend you do your own stacking code to control your folds_   
    * Time data
        * Workalendar - Business calendar for multiple countries
    * Automatic Machine Learning
        * tpot - Scikit-learn pipeline generated through genetic algorithm

* Deep Learning
    * Frameworks
        * Theano
        * Tensorflow
        * Pytorch
        * Mxnet
        * _(Not tested) Nervana Neon, Chainer, DyNet, MinPy_
    * API
        * Keras
    * Vision
        * Keras - Data augmentation
        * Scikit-image - preprocessing, segmenting (single-core)
        * Opencv - preprocessing, segmenting
    * NLP
        * spaCy - Tokenization
        * gensim - word2vec
        * ete3 - NLP trees visualization

        _NLTK is single core, extremely slow and not recommended_
    
    * Video
        * Vapoursynth - Frameserver for video pre-processing
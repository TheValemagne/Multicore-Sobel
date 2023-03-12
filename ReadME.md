# Multicore- und GPU-Computing: Projekt WS 2022-23

Implementation of horizental [sobel filter](https://en.wikipedia.org/wiki/Sobel_operator) for black and white images with use of GPU and Multicore power. This projet tests the new features of OpenMP 5.0 or higher for a project at the HTW Saar. GPU-Offloading is an experimental festure, currently supported for few c and c++ compilers ([list of OpenMP compilers](https://www.openmp.org/resources/openmp-compilers-tools/)). 

OpenMP 5 with teams and SIMD directives are not supported by MinGW or MSVC compilers on Windows.

Before to compile sobel codes, you need to install some software:
- Install OpenCV (we used vc15) on the [official web site](https://opencv.org/releases/) for Windows. We use it to load and write images.
- Set a variable with the command ``setx OpenCV_DIR {path_to}\opencv\build\x64\vc15``. This path is used in the Visual code projets and allow a ready-to-use code on every computers without changing the projets settings.
    - Update {path_to} with your OpenCV installation path and cahnge vc15 if you have installed an other version.
    - Note: undo this command with ``REG delete HKCU\Environment /F /V OpenCV_DIR``
- List of created Projets and requirements:
    - Seq-sobel and CPU-Sobel projets were setup with Visual Studio 2022 and [Intel C++ Compiler 2023](https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#dpcpp-cpp) for Teams support. Also support GPU-Offloading for Intel GPUs.
    - GPU-Sobel project run on WSL Ubuntu 22.04 LTS with g++ v11.3.0, use terminal and provided makefile. You need to enable on Windows WSL. Please read [installation.md](GPU-Sobel/installation.md) for more details.
    - Cuda-Sobel project needs [Cuda toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) and Visual Studio Build tools. Please read [installation.md](Cuda-Sobel/installation.md)
- Projects use ISO C++ 20 Standart, which provide the function std::format in c++ (not supported on GPUs)
- Images are stored in the ``images`` directory.

⚠️ Important:
- Every images generated by sobel should end with "_sobel.jpg", to be ignored by gitignore
- For new projets in visual studio:
    - Please follow this setup with $(OpenCV_DIR) : https://docs.opencv.org/4.x/dd/d6e/tutorial_windows_visual_studio_opencv.html
    - Note: in Additional Depencies only add "opencv_world460d.lib", if you use vc15 pre-build and OpenCV 4.60. The file should be in {path_to}\opencv\build\x64\vc15\lib
    - all options: floting point model to strict (OpenCV use NaN and Infinity, which generate a warning in default mode fast)
    - Update project properties for OpenMp support: [follow this tutorial](https://learn.microsoft.com/fr-fr/cpp/build/reference/openmp-enable-openmp-2-0-support?view=msvc-170)
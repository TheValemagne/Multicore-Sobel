Requirement:
- Have an Nvidia GPU, wich is compatible with CUDA
- Nvidia CUDA Toolkit, chose the right version for your GPU
- Visual Studio Build tools and add ``C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.34.31933\bin\Hostx64\x64`` to the path variables or something similar with your Visual studio and MSVC (c/c++ compiler from Microsoft) version.

Adapt paths to your envirement before lauching this command into Powershell:
- nvcc cudaSobel.cu -o cudaSobel -I"C:\Applix\opencv\build\include\" -l"C:\Applix\opencv\build\x64\vc15\lib\opencv_world460" -diag-suppress=611 
- -diag-suppress=611  suppress some warning from OpenCV files, which are here not used.

Run file:
- ./cudaSobel
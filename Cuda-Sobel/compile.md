Adapt paths to your envirement, laucht it into Powershell:

nvcc cudaSobel.cu -o cudaSobel -I"C:\Applix\opencv\build\include\" -l"C:\Applix\opencv\build\x64\vc15\lib\opencv_world460"

./cudaSobel
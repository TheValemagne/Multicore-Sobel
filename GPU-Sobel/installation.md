Setup:
- Enable [WSL 2](https://learn.microsoft.com/en-gb/windows/wsl/install) on Windows.
- [Ubuntu 22.04 LTS](https://apps.microsoft.com/store/detail/ubuntu-22042-lts/9PN20MSR04DW)
- OpenCV 4.7.0

Sources for needed packages before installation: https://pyimagesearch.com/2018/08/15/how-to-install-opencv-4-on-ubuntu/

Install Opencv on wsl: 
- sudo apt update && sudo apt install -y cmake g++ wget unzip
- sudo apt-get install libgtk-3-dev

- wget -O opencv.zip https://github.com/opencv/opencv/archive/4.7.0.zip
- unzip opencv.zip
- mv opencv-4.7.0 opencv

GCC on gpu:
- sudo apt install gcc-offload-nvptx

Create build folder:
- mkdir -p build && cd build

Before creating the build, we need a python interpreter:
- sudo pip install virtualenv virtualenvwrapper
- echo -e "\n# virtualenv and virtualenvwrapper" >> ~/.bashrc
- echo "export WORKON_HOME=$HOME/.virtualenvs" >> ~/.bashrc
- echo "export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3" >> ~/.bashrc
- echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc
- mkvirtualenv cv -p python3
- workon cv
- pip install numpy

Now back on build:
- Use: 
```
cmake -D CMAKE_BUILD_TYPE=RELEASE        -D CMAKE_INSTALL_PREFIX=/usr/local      -D INSTALL_PYTHON_EXAMPLES=ON       -D INSTALL_C_EXAMPLES=OFF       -D OPENCV_ENABLE_NONFREE=ON     -D PYTHON_EXECUTABLE=~/.virtualenvs/cv/bin/python   -D BUILD_EXAMPLES=ON ..
```

When building opencv files, you will probabely need to download  VTK and opencv_contib. VTK muss be builded and opencv_contrig be unzipped. Do it only, if the building failed and need a VTK to print "Configuration done" and "Generation done".

To solve error ``error while loading shared libraries: libopencv_highgui.so.4.4: cannot open shared object file: No such file or directory``:
- sudo touch /etc/ld.so.conf.d/opencv.conf
- sudo nano /etc/ld.so.conf.d/opencv.conf   --> open an IDE and add this content: /usr/local/lib --> ctr + o to save, ctr + x to leave
- sudo ldconfig -v

to solve errors ``
ptxas ..., line {n}; error   : Illegal operand type to instruction 'ld'
...
ptxas ..., line {m}; error   : Unknown symbol '__stack_chk_guard'
...
ptxas fatal   : Ptx assembly aborted due to errors
nvptx-as: ptxas returned 255 exit status
``
- add compiler option: -fno-stack-protector

Check that you have following folder:
- usr/local/include/opencv4/opencv2  + is not empty

Compile the code with:
```
make
```

or without makefile:
```
g++ -o gpuSobel gpuSobel.cpp -I/usr/local/include/opencv4/ -lopencv_objdetect -lopencv_features2d -lopencv_imgproc -lopencv_highgui -lopencv_core  -lopencv_imgcodecs -fopenmp -foffload=nvptx-none -fcf-protection=none  -fno-stack-protector -no-pie
```

Note:
- flag -foffload=nvptx-none is optional. The code will automatically run on NVdia GPU, wenn target regions are entered.
- Theses offload options are specific to GNU
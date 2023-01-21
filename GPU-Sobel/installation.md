Install Opencv on wsl: 
- sudo apt update && sudo apt install -y cmake g++ wget unzip
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

Solve error ``error while loading shared libraries: libopencv_highgui.so.4.4: cannot open shared object file: No such file or directory``:
- sudo touch /etc/ld.so.conf.d/opencv.conf
- sudo nano /etc/ld.so.conf.d/opencv.conf   --> open an IDE and add this content: /usr/local/lib --> ctr + o to save, ctr + x to leave
- sudo ldconfig -v

Check that you have following folder:
- usr/local/include/opencv4/ with opencv2 als folder inside

Compile the code with:
```
make gpuSobel
```

or with:
```
g++ -o gpuSobel gpuSobel.cpp -I/usr/local/include/opencv4/ -lopencv_objdetect -lopencv_features2d -lopencv_imgproc -lopencv_highgui -lopencv_core  -lopencv_imgcodecs -fopenmp -foffload=nvptx-none -fcf-protection=none  -fno-stack-protector -no-pie
```
CC = g++
OPENCV = "-I/usr/local/include/opencv4/"
OPENCV_LIBS = -lopencv_objdetect -lopencv_features2d -lopencv_imgproc -lopencv_highgui -lopencv_core -lopencv_imgcodecs
OPENMP = -fopenmp -foffload=nvptx-none -fcf-protection=none

gpuSobel : gpuSobel.cpp
	${CC} -std=c++20 -o $@ $< $(OPENCV) $(OPENCV_LIBS) $(OPENMP) -fno-stack-protector -no-pie -w

# Only -Wdeprecated-enum-enum-conversion from OpenCV
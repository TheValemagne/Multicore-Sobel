CC = g++
OPENCV = "-I/usr/local/include/opencv4/"
LIBS = -lopencv_objdetect -lopencv_features2d -lopencv_imgproc -lopencv_highgui -lopencv_core -lopencv_imgcodecs
OPENMP = -fopenmp -foffload=nvptx-none -fcf-protection=none

% : %.cpp
	${CC} -std=c++11 -o $@ $< $(opencv) $(LIBS) $(OPENMP) -fno-stack-protector -no-pie
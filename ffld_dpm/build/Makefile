CXX = g++
CFLAGS = -g -fopenmp -O3 -fPIC -I/usr/local/include/ -I/usr/include/libxml2/  -I..
LDLIBS = -g -lgomp -ljpeg -lxml2 -L/usr/local/lib64 -lfftw3f -lfftw3 -lm -Wl,-rpath,/usr/local/lib64 
OBJS = HOGPyramid.o JPEGImage.o Mixture.o Model.o Object.o Patchwork.o Rectangle.o Scene.o ffld.o
all:	ffld 

ffld: $(OBJS) 
	$(CXX) -o ffld $(OBJS) $(LDLIBS)

ffld.o: ../ffld.cpp ../HOGPyramid.h ../Intersector.h ../JPEGImage.h ../Mixture.h ../Model.h ../Object.h ../Patchwork.h ../Scene.h ../SimpleOpt.h ../Rectangle.h  
	$(CXX) $(CFLAGS) -c ../ffld.cpp

HOGPyramid.o:	../HOGPyramid.cpp ../HOGPyramid.h ../JPEGImage.h
	$(CXX) $(CFLAGS) -c ../HOGPyramid.cpp 

JPEGImage.o: ../JPEGImage.cpp ../JPEGImage.h
	$(CXX) $(CFLAGS) -c ../JPEGImage.cpp 

Mixture.o: ../Mixture.cpp ../Mixture.h ../Model.h ../Patchwork.h
	$(CXX) $(CFLAGS) -c ../Mixture.cpp 

Model.o:	../Model.cpp ../Model.h ../HOGPyramid.h
	$(CXX) $(CFLAGS) -c ../Model.cpp 

Object.o: 	../Object.cpp ../Object.h ../Rectangle.h
	$(CXX) $(CFLAGS) -c ../Object.cpp 

Patchwork.o:	../Patchwork.cpp ../Patchwork.h ../Rectangle.h ../HOGPyramid.h
	$(CXX) $(CFLAGS) -c ../Patchwork.cpp 

Rectangle.o:	../Rectangle.cpp ../Rectangle.h 
	$(CXX)	$(CFLAGS) -c ../Rectangle.cpp 

Scene.o:	../Scene.cpp ../Scene.h ../Object.h
	$(CXX)	$(CFLAGS) -c ../Scene.cpp 

clean:
	rm -f *~ *.o *.mex* *.obj 


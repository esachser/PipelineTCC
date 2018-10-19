cc = g++
saida = main.o
executavel = pipeline
boptions = -MMD -msse2 -pthread -std=c++11 -B /home/eduardo/anaconda3/compiler_compat  -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -fPIC -fopenmp
options = -MMD -msse2 -DNDEBUG -pthread -std=c++11 -O3 -Wl,--no-as-needed  -fopenmp
libs_opencv := $(shell pkg-config --libs-only-l ~/anaconda3/lib/pkgconfig/opencv.pc)
MKLROOT = /opt/intel/mkl
# libs = -L${MKLROOT}/lib -lstdc++ -lmkl_rt -liomp5 -lm -ldl -L/usr/local/lib $(libs_opencv)
libs_mkl = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group
libs = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lstdc++ -liomp5 -lm -ldl -L/usr/local/lib $(libs_opencv)
includes = -I. -I./src -Ilib/spams_lib/spams/linalg -Ilib/spams_lib/spams/prox -Ilib/spams_lib/spams/decomp -Ilib/spams_lib/spams/dictLearn -I/usr/local/include -I/usr/include -I/home/eduardo/anaconda3/include -c
includes2 = -I. -I./src -I/opt/intel/mkl/include -I/usr/local/cuda/include -I/usr/local/include -I/usr/include -c

build:
	$(cc) $(boptions) $(includes) src/main.cpp
	$(cc) $(options) $(libs) main.o -o $(executavel)

buildcreate:
	$(cc) $(boptions) $(includes) src/createframes.cpp
	$(cc) $(options) $(libs) createframes.o -o createframes
	./createframes

buildmain2:
	$(cc) $(boptions) $(includes2) src/main2.cpp
	$(cc) $(options) $(libs) dylink_nvcuvid.o main2.o -o eigensolution

buildcvcuda:
	$(cc) $(boptions) $(includes2) src/cvcuda.cpp
	$(cc) $(options) $(libs) -lopencv_cudacodec  cvcuda.o -o cvcuda

buildsep:
	$(cc) $(boptions) $(includes) src/mainsep.cpp
	$(cc) $(options) $(libs) mainsep.o -o $(executavel)sep

clean:
	rm main.o $(executavel)

run: build
	./$(executavel)

all: build
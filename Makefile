cc = g++
saida = main.o
executavel = pipeline
boptions = -MMD -msse2 -pthread -std=c++11 -B /home/eduardo/anaconda3/compiler_compat  -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -fPIC -fopenmp
options = -MMD -msse2 -DNDEBUG -pthread -std=c++11 -O3 -B /home/eduardo/anaconda3/compiler_compat -L/home/eduardo/anaconda3/lib -Wl,-rpath=/home/eduardo/anaconda3/lib -Wl,--no-as-needed  -fopenmp
libs_opencv := $(shell pkg-config --cflags --libs ~/anaconda3/lib/pkgconfig/opencv.pc)
libs = -L/home/eduardo/anaconda3/lib -L/opt/intel/mkl/lib  -lstdc++ -lmkl_rt -liomp5 -lm -ldl $(libs_opencv)
MKLROOT = /opt/intel/mkl
libs_mkl = ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group
includes = -I. -I./src -Ilib/spams_lib/spams/linalg -Ilib/spams_lib/spams/prox -Ilib/spams_lib/spams/decomp -Ilib/spams_lib/spams/dictLearn -I/usr/local/include -I/usr/include -I/home/eduardo/anaconda3/include -c
includes2 = -I. -I./src -I/opt/intel/mkl/include -I/usr/local/include -I/usr/include -I/home/eduardo/anaconda3/include -c

build:
	$(cc) $(boptions) $(includes) src/main.cpp
	$(cc) $(options) $(libs) main.o -o $(executavel)

buildcreate:
	$(cc) $(boptions) $(includes) src/createframes.cpp
	$(cc) $(options) $(libs) createframes.o -o createframes
	./createframes

buildmain2:
	$(cc) $(boptions) -DMKL_LP64 $(includes2) src/main2.cpp
	$(cc) $(options) -DMKL_ILP64 $(libs) main2.o -o eigensolution

buildsep:
	$(cc) $(boptions) $(includes) src/mainsep.cpp
	$(cc) $(options) $(libs) mainsep.o -o $(executavel)sep

clean:
	rm main.o $(executavel)

run: build
	./$(executavel)

all: build
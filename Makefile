cc = g++
saida = main.o
executavel = pipeline
boptions = -pthread -std=c++11 -B /home/eduardo/anaconda3/compiler_compat  -Wsign-compare -DNDEBUG -g -fwrapv -Ofast -Wall -fPIC -fopenmp
options = -pthread -std=c++11 -Ofast -B /home/eduardo/anaconda3/compiler_compat -L/home/eduardo/anaconda3/lib -Wl,-rpath=/home/eduardo/anaconda3/lib -Wl,--no-as-needed  -fopenmp
libs_opencv := $(shell pkg-config --cflags --libs ~/anaconda3/lib/pkgconfig/opencv.pc)
libs = -L/home/eduardo/anaconda3/lib -lstdc++ -lmkl_rt -liomp5 $(libs_opencv)
includes = -I. -Ilib/spams_lib/spams/linalg -Ilib/spams_lib/spams/prox -Ilib/spams_lib/spams/decomp -Ilib/spams_lib/spams/dictLearn -I/usr/local/include -I/usr/include -I/home/eduardo/anaconda3/include -c

build:
	$(cc) $(boptions) $(includes) src/main.cpp
	$(cc) $(options) $(libs) main.o -o $(executavel)

buildcreate:
	$(cc) $(boptions) $(includes) src/createframes.cpp
	$(cc) $(options) $(libs) createframes.o -o createframes
	./createframes

clean:
	rm main.o $(executavel)

run: build
	./$(executavel)

all: build
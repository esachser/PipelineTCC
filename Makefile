cc = g++
saida = main.o
executavel = pipeline
boptions = -MMD -msse2 -pthread -std=c++11 -B /home/eduardo/anaconda3/compiler_compat  -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -fPIC -fopenmp
options = -MMD -msse2 -DNDEBUG -pthread -std=c++11 -O3 -fopenmp
libs_opencv := $(shell pkg-config --libs-only-l ~/anaconda3/lib/pkgconfig/opencv.pc)
MKLROOT = /opt/intel/mkl
# libs = -L${MKLROOT}/lib -lstdc++ -lmkl_rt -liomp5 -lm -ldl -L/usr/local/lib $(libs_opencv)
libs_mkl = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group
libs = -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lstdc++ -liomp5 -lm -ldl -L/usr/local/lib $(libs_opencv)
includes = -I. -I./src -Ilib/spams_lib/spams/linalg -Ilib/spams_lib/spams/prox -Ilib/spams_lib/spams/decomp -Ilib/spams_lib/spams/dictLearn -I/usr/local/include -I/usr/include -I/home/eduardo/anaconda3/include -c
includes2 = -I. -I./src -I/opt/intel/mkl/include -I/usr/local/cuda/include -I/usr/local/include -I/usr/include -c


# Compilação usando CUDA
CUDA_PATH ?= "/usr/local/cuda-9.2"
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(cc)
NVCCFLAGS   := -m64 --default-stream per-thread

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(options))

SAMPLE_ENABLED := 1

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
LIBRARIES_CUDA := $(libs) -lopencv_cudacodec

# Gencode arguments
SMS ?= 60

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

ALL_CCFLAGS += -dc -O3
LIBRARIES_CUDA += -lcublas -lcublas_device -lcudadevrt

build:
	$(cc) $(boptions) $(includes) src/main.cpp
	$(cc) $(options) $(libs) main.o -o $(executavel)

buildcreate:
	$(cc) $(options) $(includes2) src/createframes.cpp
	$(cc) $(options) createframes.o -o createframes $(libs)
	rm -rf ../trainframes/frame_*
	./createframes

buildmain2:
	$(cc) $(options) $(includes2) src/main2.cpp
	$(cc) $(options) main2.o -o eigensolution $(libs)

buildcvcuda:
	$(NVCC) $(includes2) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o kernels.o -c ./src/kernels.cu
	$(NVCC) $(includes2) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o cvcuda.o -c ./src/cvcuda.cpp
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o cvcudaeigensolution kernels.o cvcuda.o $(LIBRARIES_CUDA)

buildsep:
	$(cc) $(boptions) $(includes) src/mainsep.cpp
	$(cc) $(options) $(libs) mainsep.o -o $(executavel)sep

buildcuda:
	$(NVCC) $(includes2) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o kernels.o -c ./src/kernels.cu
	$(NVCC) $(includes2) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o main2.o -c ./src/main2.cpp
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o cudaeigensolution kernels.o main2.o $(LIBRARIES_CUDA)

clean:
	rm main.o $(executavel)

run: build
	./$(executavel)

all: build
#Project structure
INCDIR=inc
SRCDIR=src
OBJDIR=obj
COBJDIR=cobj
TESTDIR=test
UNITTESTDIR=$(TESTDIR)/unitTest
PERMATESTDIR=$(TESTDIR)/permaTest
COLORPERMATESTDIR=$(TESTDIR)/colorPermaTest
NETTESTDIR=$(TESTDIR)/netTest
DOWNLOADSDIR=$(NETTESTDIR)/downloaded
PERFTESTDIR=$(TESTDIR)/perfTest
CUPERFTESTDIR=$(TESTDIR)/cuPerfTest
COLORPERFTESTDIR=$(TESTDIR)/colorPerfTest
3RDDIR=3rdparty

#G++ compiler settings
CXX=g++
CXXINC=-I ./$(INCDIR)/
CXXFLAGS=-O3 -std=c++11 $(CXXINC) -g -rdynamic -Werror=return-type

INCFILES=$(wildcard $(INCDIR)/*.h)
SRCFILES=$(wildcard $(SRCDIR)/*.cpp)
CUSRCFILES=$(wildcard $(SRCDIR)/*.cu)
CSRCFILES=$(wildcard $(SRCDIR)/*.c)
TESTFILES=$(wildcard $(TESTDIR)/*/*.cpp)

#GCC compiler settings
GCC=gcc
GCCLIBS=-L./3rdparty/Csdp/lib -lsdp -llapack -lblas -lgfortran -lquadmath -lm -lgomp -pthread -ldl
GCCFLAGS=-m64 -march=native -mtune=native -Ofast -fopenmp -ansi -Wall -DBIT64 -DUSEOPENMP -DSETNUMTHREADS -DUSESIGTERM -DUSEGETTIME -I./3rdparty/Csdp/include

# CUDA compiler settings
CUDA=/usr/local/cuda
NVCC=$(CUDA)/bin/nvcc
NVCCSM=sm_37
NVCCINC=-I $(CUDA)/include \
		-I $(CUDA)/samples/common/inc \
		-I ./$(3RDDIR)/moderngpu/src \
		-I ./$(INCDIR)/ 
NVCCFLAGS=-arch $(NVCCSM) -O3 -std=c++11 --expt-extended-lambda -w $(NVCCINC)
LDFLAGS=-L/usr/local/cuda/lib64 -lcudart

OBJFILES=$(patsubst %.cpp, $(OBJDIR)/%.o, $(SRCFILES)) \
				 $(patsubst %.cu, $(OBJDIR)/%.o, $(CUSRCFILES))
COBJFILES=$(patsubst %.c, $(COBJDIR)/%.o, $(CSRCFILES))
UNITTESTFILES=$(patsubst %.cpp, $(OBJDIR)/%.e, $(wildcard $(UNITTESTDIR)/*.cpp)) \
						  $(patsubst %.cu, $(OBJDIR)/%.e, $(wildcard $(UNITTESTDIR)/*.cu)) \
						  
PERMATESTFILES=$(patsubst %.cu, $(OBJDIR)/%.e, $(wildcard $(PERMATESTDIR)/*.cu))
COLORPERMATESTFILES=$(patsubst %.cpp, $(OBJDIR)/%.e, $(wildcard $(COLORPERMATESTDIR)/*.cpp))
NETTESTFILES=$(patsubst %.cpp, $(OBJDIR)/%.e, $(wildcard $(NETTESTDIR)/*.cpp))
PERFTESTFILES=$(patsubst %.cpp, $(OBJDIR)/%.e, $(wildcard $(PERFTESTDIR)/*.cpp))
CUPERFTESTFILES=$(patsubst %.cu, $(OBJDIR)/%.e, $(wildcard $(CUPERFTESTDIR)/*.cu)) \
							  $(patsubst %.cpp, $(OBJDIR)/%.e, $(wildcard $(CUPERFTESTDIR)/*.cpp))
COLORPERFTESTFILES=$(patsubst %.cpp, $(OBJDIR)/%.e, $(wildcard $(COLORPERFTESTDIR)/*.cpp))

# $(info $(OBJFILES))
# $(info $(TESTFILES))
# $(info $(PERMATESTFILES))

#Targets
# test: $(TESTFILES)
unitTest: $(UNITTESTFILES) $(COBJFILES)
permaTest: $(PERMATESTFILES)
colorPermaTest: $(COLORPERMATESTFILES)
netTest: $(NETTESTFILES)
perfTest: $(PERFTESTFILES)
cuPerfTest: $(CUPERFTESTFILES)
colorPerfTest: $(COLORPERFTESTFILES)

$(COBJDIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(GCC) $(GCCFLAGS) -c $< $(GCCLIBS) -o $@

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR)/%.e: %.cpp $(OBJFILES) $(COBJFILES)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OBJFILES) $(COBJFILES) $< $(GCCLIBS) -o $@ $(LDFLAGS)

$(OBJDIR)/%.e: %.cu obj/src/perfect.o  obj/src/jewels.o  obj/src/pyramids.o  obj/src/oddHoles.o  obj/src/commons.o  obj/src/testCommons.o  obj/src/cuCommons.o  obj/src/nearCleaners.o 
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) obj/src/perfect.o  obj/src/jewels.o  obj/src/pyramids.o  obj/src/oddHoles.o  obj/src/commons.o  obj/src/testCommons.o  obj/src/cuCommons.o  obj/src/nearCleaners.o $<  -o $@ $(LDFLAGS)
  

.PHONY: test clean run-tests unitTest retest netTest lint

# Do not delete OBJFILES by default (shortens make time)
.OBJ: $(OBJFILES)

unitTest: 
	./test/unitTest/runUnitTest.sh

retest: clean unitTest

permaTest: 
	./obj/test/permaTest/perfect.e

colorPermaTest: 
	./obj/test/colorPermaTest/color.e

netTest:
	./test/netTest/prepare.sh

perfTest:
	./test/perfTest/run.sh

cuPerfTest:
	./test/cuPerfTest/run.sh

colorPerfTest:
	./test/colorPerfTest/run.sh

clean:
	rm -rf obj cobj *.e

cleanDownloads:
	rm $(DOWNLOADSDIR) -rf

theta: ./cobj/src/theta.o

lint:
	~/.local/bin/cpplint --linelength=120 --extensions=cpp,h --filter=-legal/copyright,-runtime/references,-build/include_subdir,-readability/multiline_string $(SRCFILES) $(INCFILES) $(TESTFILES)

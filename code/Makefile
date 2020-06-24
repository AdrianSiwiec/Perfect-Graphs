#Project structure
INCDIR=inc
SRCDIR=src
OBJDIR=obj
TESTDIR=test
UNITTESTDIR=$(TESTDIR)/unitTest
PERMATESTDIR=$(TESTDIR)/permaTest
NETTESTDIR=$(TESTDIR)/netTest
DOWNLOADSDIR=$(NETTESTDIR)/downloaded
PERFTESTDIR=$(TESTDIR)/perfTest

#GCC compiler settings
CXX=g++
CXXINC=-I ./$(INCDIR)/
CXXFLAGS=-O3 -std=c++11 $(CXXINC) -g -rdynamic -Werror=return-type

INCFILES=$(wildcard $(INCDIR)/*.h)
SRCFILES=$(wildcard $(SRCDIR)/*.cpp)
TESTFILES=$(wildcard $(TESTDIR)/*/*.cpp)

OBJFILES=$(patsubst %.cpp, $(OBJDIR)/%.o, $(SRCFILES))
UNITTESTFILES=$(patsubst %.cpp, $(OBJDIR)/%.e, $(wildcard $(UNITTESTDIR)/*.cpp))
PERMATESTFILES=$(patsubst %.cpp, $(OBJDIR)/%.e, $(wildcard $(PERMATESTDIR)/*.cpp))
NETTESTFILES=$(patsubst %.cpp, $(OBJDIR)/%.e, $(wildcard $(NETTESTDIR)/*.cpp))
PERFTESTFILES=$(patsubst %.cpp, $(OBJDIR)/%.e, $(wildcard $(PERFTESTDIR)/*.cpp))


# $(info $(OBJFILES))
# $(info $(TESTFILES))
# $(info $(PERMATESTFILES))

#Targets
# test: $(TESTFILES)
unitTest: $(UNITTESTFILES)
permaTest: $(PERMATESTFILES)
netTest: $(NETTESTFILES)
perfTest: $(PERFTESTFILES)

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/%.e: %.cpp $(OBJFILES) 
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OBJFILES) $< -o $@

.PHONY: test clean run-tests unitTest retest netTest lint

# Do not delete OBJFILES by default (shortens make time)
.OBJ: $(OBJFILES)

unitTest: 
	./test/unitTest/runUnitTest.sh

retest: clean unitTest

permaTest: 
	./obj/test/permaTest/perfect.e

netTest:
	./test/netTest/prepare.sh

perfTest:
	./test/perfTest/run.sh

clean:
	rm -rf obj *.e

cleanDownloads:
	rm $(DOWNLOADSDIR) -rf

lint:
	~/.local/bin/cpplint --linelength=120 --extensions=cpp,h --filter=-legal/copyright,-runtime/references,-build/include_subdir,-readability/multiline_string $(SRCFILES) $(INCFILES) $(TESTFILES)

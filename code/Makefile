#Project structure
INCDIR=inc
SRCDIR=src
OBJDIR=obj
TESTDIR=test
UNITTESTDIR=$(TESTDIR)/unitTest
PERMATESTDIR=$(TESTDIR)/permaTest
NETTESTDIR=$(TESTDIR)/netTest
DOWNLOADSDIR=$(NETTESTDIR)/downloaded

#GCC compiler settings
CXX=g++
CXXINC=-I ./$(INCDIR)/
CXXFLAGS=-O2 -std=c++11 $(CXXINC) -g -rdynamic -Werror=return-type

OBJFILES=$(patsubst %.cpp, $(OBJDIR)/%.o, $(wildcard $(SRCDIR)/*.cpp))
TESTFILES=$(patsubst %.cpp, $(OBJDIR)/%.e, $(wildcard $(TESTDIR)/*.cpp))
UNITTESTFILES=$(patsubst %.cpp, $(OBJDIR)/%.e, $(wildcard $(UNITTESTDIR)/*.cpp))
PERMATESTFILES=$(patsubst %.cpp, $(OBJDIR)/%.e, $(wildcard $(PERMATESTDIR)/*.cpp))
NETTESTFILES=$(patsubst %.cpp, $(OBJDIR)/%.e, $(wildcard $(NETTESTDIR)/*.cpp))


# $(info $(OBJFILES))
# $(info $(TESTFILES))
# $(info $(PERMATESTFILES))

#Targets
test: $(TESTFILES)
unitTest: $(UNITTESTFILES)
permaTest: $(PERMATESTFILES)
netTest: $(NETTESTFILES)

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/%.e: %.cpp $(OBJFILES) 
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OBJFILES) $< -o $@

.PHONY: test clean run-tests unitTest retest netTest

# Do not delete OBJFILES by default (shortens make time)
.OBJ: $(OBJFILES)

unitTest: 
	./test/unitTest/runUnitTest.sh

retest: clean unitTest

permaTest: 
	./obj/test/permaTest/perfect.e

netTest:
	./test/netTest/prepare.sh

clean:
	rm -rf obj *.e

cleanDownloads:
	rm $(DOWNLOADSDIR) -rf

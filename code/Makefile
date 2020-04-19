#Project structure
INCDIR=inc
SRCDIR=src
OBJDIR=obj
TESTDIR=test
PERMATESTDIR=permaTest

#GCC compiler settings
CXX=g++
CXXINC=-I ./$(INCDIR)/
CXXFLAGS=-O2 -std=c++11 $(CXXINC) -g -rdynamic -Werror=return-type

OBJFILES=$(patsubst %.cpp, $(OBJDIR)/%.o, $(wildcard $(SRCDIR)/*.cpp))
TESTFILES=$(patsubst %.cpp, $(OBJDIR)/%.e, $(wildcard $(TESTDIR)/*.cpp))
PERMATESTFILES=$(patsubst %.cpp, $(OBJDIR)/%.e, $(wildcard $(PERMATESTDIR)/*.cpp))

$(info $(OBJFILES))
$(info $(TESTFILES))
$(info $(PERMATESTFILES))

#Targets
test: $(TESTFILES)
permaTest: $(PERMATESTFILES)

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/%.e: %.cpp $(OBJFILES) 
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OBJFILES) $< -o $@

.PHONY: test clean run-tests retest rt

# Do not delete OBJFILES by default (shortens make time)
.OBJ: $(OBJFILES)

run-tests: test
	for test in obj/test/*.e; do \
		echo "Running $$test" && \
		./$$test || exit; \
	done &&	echo "ALL OK"

retest: clean run-tests

permaTest: test
	./obj/permaTest/perfect.e

clean:
	rm -rf obj *.e

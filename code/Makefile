#Project structure
INCDIR=inc
SRCDIR=src
OBJDIR=obj
TESTDIR=test

#GCC compiler settings
CXX=g++
CXXINC=-I ./$(INCDIR)/
CXXFLAGS=-O2 -std=c++11 $(CXXINC) -g -rdynamic -Werror=return-type

INCFILES=$(wildcard $(INCDIR)/*.h)
SRCFILES=$(wildcard $(SRCDIR)/*.cpp)
OBJFILES=$(patsubst %.cpp, $(OBJDIR)/%.o, $(SRCFILES))
TESTFILES=$(patsubst %.cpp, $(OBJDIR)/%.e, $(wildcard $(TESTDIR)/*.cpp))

$(info $(OBJFILES))
$(info $(TESTFILES))

#Targets
test: $(TESTFILES)

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/%.e: %.cpp $(OBJFILES) 
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OBJFILES) $< -o $@

.PHONY: test clean run-tests retest lint

# Do not delete OBJFILES by default (shortens make time)
.OBJ: $(OBJFILES)

run-tests: test
	for test in obj/test/*.e; do \
		echo "Running $$test" && \
		./$$test || exit; \
	done &&	echo "ALL OK"

retest: clean run-tests

clean:
	rm -rf obj *.e

lint:
	cpplint --linelength=120 --extensions=cpp,h --filter=-legal/copyright,-runtime/references $(SRCFILES) $(INCFILES)

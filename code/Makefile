#Project structure
INCDIR=inc
SRCDIR=src
OBJDIR=obj
TESTDIR=test

#GCC compiler settings
CXX=g++
CXXINC=-I ./$(INCDIR)/
CXXFLAGS=-O2 -std=c++11 $(CXXINC)

OBJFILES=$(patsubst %.cpp, obj/%.o, $(wildcard src/*.cpp))



#Targets
test: $(OBJDIR)/$(TESTDIR)/diamonds.e

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(CXXINC) -c $< -o $@

$(OBJDIR)/%.e: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(CXXINC) -c $< -o $@

.PHONY: test clean

clean:
	rm -rf obj *.e

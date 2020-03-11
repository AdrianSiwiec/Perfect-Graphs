#pragma once

#include "commons.h"
#include <cassert>
#include <cstdlib>
#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

bool probTrue(double p);
void printGraph(const Graph &G);
Graph getRandomGraph(int size, double p);
vec<Graph> getRandomGraphs(int size, double p, int howMany);

bool isAllZeros(vec<int> v);
vec<int> nextTuple(vec<int> v, int max);
vec<vec<int>> generateTuples(int size, int max);

void handler(int sig);
void init();
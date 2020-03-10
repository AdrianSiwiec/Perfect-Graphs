#pragma once

#include "commons.h"
#include <cassert>
#include <cstdlib>

bool probTrue(double p);
void printGraph(const Graph &G);
Graph getRandomGraph(int size, double p);
vec<Graph> getRandomGraphs(int size, double p, int howMany);
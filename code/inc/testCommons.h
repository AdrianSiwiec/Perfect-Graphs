#pragma once

#include "commons.h"
#include <cassert>
#include <cstdlib>
#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// Whether to run big tests. These take more time.
const bool bigTests = false;
// const bool bigTests = true;

bool probTrue(double p);
void printGraph(const Graph &G);
Graph getRandomGraph(int size, double p);
vec<Graph> getRandomGraphs(int size, double p, int howMany);

void handler(int sig);
void init(bool srandTime = false);
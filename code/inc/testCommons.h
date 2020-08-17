#pragma once

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <map>
#include <random>
#include <string>
#include "commons.h"

#ifndef __CYGWIN__
#include <execinfo.h>
#endif

using std::get;
using std::invalid_argument;
using std::map;
using namespace std::chrono;

// Whether to run big tests. These take more time.
const bool bigTests = false;  // TODO(Adrian) make bigTests big again (perf)
// const bool bigTests = true;
const int _max_threads_to_run = 10000;

bool probTrue(double p);
void printGraph(const Graph &G);
Graph getRandomGraph(int size, double p);
// will loop when impossible or unlikely.
Graph getRandomPerfectGraph(int size, double p);
vec<Graph> getRandomGraphs(int size, double p, int howMany);

Graph getNonPerfectGraph(int holeSize, int reminderSize, double p);

// Returns biparite graph, two equal layers (+-1), each edge between layers has probability of p.
Graph getBipariteGraph(int size, double p);

void handler(int sig);
void init(bool srandTime = false);

struct RaiiTimer {
  explicit RaiiTimer(string msg = "");
  ~RaiiTimer();

  double getElapsedSeconds();

 private:
  nanoseconds start_ns;
  string msg;
};

bool testWithStats(const Graph &G, bool naive);
void printStats();

double getDistr();
void testGraph(const Graph &G, bool verbose);
void testGraph(const Graph &G, bool result, bool verbose);

void printTimeHumanReadable(int64_t time);

struct RaiiProgressBar {
  explicit RaiiProgressBar(int allTests);
  ~RaiiProgressBar();

  void update(int testsDone);

 private:
  int allTests;
  nanoseconds start_ns;
  const int width = 80;

  int getFilled(int testsDone);
};

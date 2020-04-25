#pragma once

#include "commons.h"
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <execinfo.h>
#include <map>
#include <random>
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

Graph getNonPerfectGraph(int holeSize, int reminderSize, double p);

// Returns biparite graph, two equal layers (+-1), each edge between layers has probability of p.
Graph getBipariteGraph(int size, double p);

void handler(int sig);
void init(bool srandTime = false);

struct RaiiTimer {
  RaiiTimer(string msg);
  ~RaiiTimer();

private:
  clock_t startTimer;
  string msg;
};

bool testWithStats(const Graph &G, bool naive);
void printStats();
double getDistr();
void testGraph(const Graph &G, bool verbose);
void testGraph(const Graph &G, bool result, bool verbose);

void printTimeHumanReadable(long long time);

struct RaiiProgressBar {
  RaiiProgressBar(int allTests);
  ~RaiiProgressBar();

  void update(int testsDone);

private:
  int allTests;
  clock_t startTimer;
  const int width = 80;

  int getFilled(int testsDone);
};
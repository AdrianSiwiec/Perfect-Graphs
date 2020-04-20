#include "testCommons.h"
#include "commons.h"
#include <cstdlib>
#include <iostream>

bool probTrue(double p) { return rand() / (RAND_MAX + 1.0) < p; }

void printGraph(const Graph &G) {
  cout << "   ";
  for (int i = 0; i < G.n; i++) {
    cout << i;
    if (i < 10)
      cout << " ";
  }
  cout << endl;
  for (int i = 0; i < G.n; i++) {
    cout << i;
    cout << ":";
    if (i < 10)
      cout << " ";
    for (int j = 0; j < G.n; j++) {
      cout << (G.areNeighbours(i, j) ? "X " : ". ");
    }
    cout << endl;
  }
  cout << endl;
}

Graph getRandomGraph(int size, double p) {
  vec<vec<int>> neighbours(size);
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (probTrue(p)) {
        neighbours[i].push_back(j);
        neighbours[j].push_back(i);
      }
    }
  }

  return Graph(neighbours);
}

vec<Graph> getRandomGraphs(int size, double p, int howMany) {
  vec<Graph> ret;
  ret.reserve(howMany);
  for (int i = 0; i < howMany; i++) {
    ret.push_back(getRandomGraph(size, p));
  }

  return ret;
}

Graph getNonPerfectGraph(int holeSize, int reminderSize, double p) {
  if (holeSize < 5 || holeSize % 2 == 0) {
    throw invalid_argument("Trying to generate Non perfect graph with incorrect hole size.");
  }

  int size = holeSize + reminderSize;
  vec<vec<int>> neighbours(size);

  for (int i = 1; i < holeSize; i++) {
    neighbours[i - 1].push_back(i);
    neighbours[i].push_back(i - 1);
  }
  neighbours[0].push_back(holeSize - 1);
  neighbours[holeSize - 1].push_back(0);

  Graph reminder = getRandomGraph(reminderSize, p);
  for (int i = 0; i < reminderSize; i++) {
    for (int v : reminder[i]) {
      neighbours[i + holeSize].push_back(v + holeSize);
    }
  }

  for (int i = 0; i < holeSize; i++) {
    for (int j = holeSize; j < size; j++) {
      if (probTrue(p)) {
        neighbours[i].push_back(j);
        neighbours[j].push_back(i);
      }
    }
  }

  return Graph(neighbours).getShuffled();
}

Graph getBipariteGraph(int size, double p) {
  vec<vec<int>> neighbours(size);
  for (int i = 0; i < size / 2; i++) {
    for (int j = size / 2; j < size; j++) {
      if (probTrue(p)) {
        neighbours[i].push_back(j);
        neighbours[j].push_back(i);
      }
    }
  }

  return Graph(neighbours).getShuffled();
}

void handler(int sig) {
  void *array[100];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, 100);

  // print out all the frames to stderr
  fprintf(stderr, "Error: signal %d:\n", sig);
  auto messages = backtrace_symbols(array, size);

  /* skip first stack frame (points here) */
  for (int i = 1; i < size && messages != NULL; ++i) {
    if ((messages[i][1] != 'l' || messages[i][2] != 'i' || messages[i][3] != 'b') &&
        (messages[i][1] != 'u' || messages[i][2] != 's' || messages[i][3] != 'r'))
      fprintf(stderr, "\t[bt]: (%d) %s\n", i, messages[i]);
  }

  exit(1);
}

void init(bool srandTime) {
  if (srandTime) {
    srand(time(NULL));
  }

  signal(SIGSEGV, handler);
  signal(SIGABRT, handler);
}

RaiiTimer::RaiiTimer(string msg) : msg(msg) { startTimer = clock(); }
RaiiTimer::~RaiiTimer() {
  double duration = (clock() - startTimer) / (double)CLOCKS_PER_SEC;
  cout << msg << ": " << duration << "s" << endl;
}
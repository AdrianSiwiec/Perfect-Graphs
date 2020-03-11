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
bool isAllZeros(vec<int> v) {
  for (int a : v)
    if (a != 0)
      return false;
  return true;
}

vec<int> nextTuple(vec<int> v, int max) {
  v[0]++;
  for (int i = 0; i < v.size() && v[i] >= max; i++) {
    v[i] = 0;
    if (i + 1 < v.size())
      v[i + 1]++;
  }

  return v;
}

vec<vec<int>> generateTuples(int size, int max) {
  vec<vec<int>> ret;
  vec<int> current = vec<int>(size);
  do {
    ret.push_back(current);
    current = nextTuple(current, max);
  } while (!isAllZeros(current));

  return ret;
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

void init() {
  signal(SIGSEGV, handler);
  signal(SIGABRT, handler);
}
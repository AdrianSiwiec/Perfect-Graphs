#include "testCommons.h"

#include <chrono>
#include <cstdlib>
#include <iostream>

#include "color.h"
#include "commons.h"
#include "oddHoles.h"
#include "perfect.h"

using namespace std::chrono;
using std::default_random_engine;
using std::flush;
using std::make_pair;
using std::normal_distribution;

bool probTrue(double p) { return rand() / (RAND_MAX + 1.0) < p; }

void printGraph(const Graph &G) {
  cout << "   ";
  for (int i = 0; i < G.n; i++) {
    cout << i;
    if (i < 10) cout << " ";
  }
  cout << endl;
  for (int i = 0; i < G.n; i++) {
    cout << i;
    cout << ":";
    if (i < 10) cout << " ";
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

Graph getRandomPerfectGraph(int size, double p) {
  Graph G(0);
  do {
    G = getRandomGraph(size, p);
  } while (!isPerfectGraph(G));

  return G;
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

#ifdef __CYGWIN__
  fprintf(stderr, "Error: signal %d:\n", sig);
#else
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
#endif

  exit(1);
}

void init(bool srandTime) {
  if (srandTime) {
    srand(time(NULL));
  }

  signal(SIGSEGV, handler);
  signal(SIGABRT, handler);
}

RaiiTimer::RaiiTimer(string msg) : msg(msg) {
  start_ns = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
}
RaiiTimer::~RaiiTimer() {
  nanoseconds end_ns = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
  double duration = (end_ns.count() - start_ns.count()) / 1e9;
  if (msg.size() > 0) cout << msg << ": " << duration << "s" << endl;
}
double RaiiTimer::getElapsedSeconds() {
  nanoseconds end_ns = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
  double duration = (end_ns.count() - start_ns.count()) / 1e9;
  return duration;
}

map<pair<int, bool>, double> sumTime;
map<pair<int, bool>, double> sumClockTime;
map<pair<int, bool>, int> casesTested;

map<pair<int, bool>, double> sumTimeNaive;
map<pair<int, bool>, double> sumClockTimeNaive;
map<pair<int, bool>, int> casesTestedNaive;

default_random_engine generator;
normal_distribution<double> distribution(0.5, 0.15);

bool testWithStats(const Graph &G, bool naive) {
  nanoseconds start_ns = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
  clock_t start_clock = clock();

  bool result = naive ? isPerfectGraphNaive(G) : isPerfectGraph(G);

  nanoseconds end_ns = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
  clock_t end_clock = clock();

  double duration = (end_ns.count() - start_ns.count()) / 1e9;
  double clock_duration = (end_clock - start_clock) / static_cast<double>(CLOCKS_PER_SEC);

  if (naive) {
    sumTimeNaive[make_pair(G.n, result)] += duration;
    sumClockTimeNaive[make_pair(G.n, result)] += clock_duration;
    casesTestedNaive[make_pair(G.n, result)]++;
  } else {
    sumTime[make_pair(G.n, result)] += duration;
    sumClockTime[make_pair(G.n, result)] += clock_duration;
    casesTested[make_pair(G.n, result)]++;
  }

  return result;
}

void printStats() {
  if (!sumTimeNaive.empty()) cout << "Naive recognition stats: " << endl;
  for (auto it = sumTimeNaive.begin(); it != sumTimeNaive.end(); it++) {
    int cases = casesTestedNaive[it->first];
    cout << "\tn=" << it->first.first << ", result=" << it->first.second << ", cases=" << cases
         << ", avgTime=" << it->second / cases
         << ", parallel factor=" << sumClockTimeNaive[it->first] / it->second << endl;
  }
  if (!sumTime.empty()) cout << "Perfect recognition stats: " << endl;
  for (auto it = sumTime.begin(); it != sumTime.end(); it++) {
    if (!it->first.second) continue;
    int cases = casesTested[it->first];
    cout << "\tn=" << it->first.first << ", result=" << it->first.second << ", cases=" << cases
         << ", avgTime=" << it->second / cases << ", parallel factor=" << sumClockTime[it->first] / it->second
         << endl;
  }
  for (auto it = sumTime.begin(); it != sumTime.end(); it++) {
    if (it->first.second) continue;
    int cases = casesTested[it->first];
    cout << "\tn=" << it->first.first << ", result=" << it->first.second << ", cases=" << cases
         << ", avgTime=" << it->second / cases << ", parallel factor=" << sumClockTime[it->first] / it->second
         << endl;
  }
}

double getDistr() {
  double distr = distribution(generator);
  if (distr <= 0 || distr > 1) distr = 0.5;

  return distr;
}

void testGraph(const Graph &G, bool verbose) {
  if (verbose) cout << "Testing " << G.n << " vs naive" << endl;

  bool naivePerfect = testWithStats(G, true);

  bool perfect = testWithStats(G, false);

  if (naivePerfect != perfect) {
    cout << "ERROR: " << endl << "naive=" << naivePerfect << endl << "perfect=" << perfect << endl;
    cout << G << endl;
    if (!naivePerfect) cout << findOddHoleNaive(G) << endl;
  }

  assert(naivePerfect == perfect);
}

void testGraph(const Graph &G, bool result, bool verbose) {
  if (verbose) cout << "Testing " << G.n << " vs " << result << endl;

  bool perfect = testWithStats(G, false);

  if (perfect != result) {
    cout << "Error Test Graph" << endl;
    cout << G << endl;
    cout << "Expected " << result << ", got " << perfect << endl;
  }

  assert(perfect == result);
}

void printTimeHumanReadable(double time) {
  double s = time;
  int h = s / (60 * 60);
  s -= h * (60 * 60);
  if (h != 0) {
    cout << h << "h";
  }

  int m = s / 60;
  s -= m * 60;
  if (m != 0) {
    cout << m << "m";
  }

  cout << static_cast<int>(s) + 1 << "s";
}

RaiiProgressBar::RaiiProgressBar(int allTests) : allTests(allTests) {
  start_ns = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
  update(0);
}

RaiiProgressBar::~RaiiProgressBar() { cout << endl; }

int RaiiProgressBar::getFilled(int testsDone) {
  double progress = testsDone / (static_cast<double>(allTests));
  return width * progress;
}

void RaiiProgressBar::update(int testsDone) {
  int toFill = getFilled(testsDone);
  if (testsDone < 10 || testsDone == allTests || toFill != getFilled(testsDone - 1)) {
    cout << "[";
    for (int i = 0; i < width; i++) {
      cout << (i < toFill ? "X" : " ");
    }
    cout << "]";
    if (testsDone > 0) {
      cout << " (about ";
      nanoseconds end_ns = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
      double timeElapsed = (end_ns.count() - start_ns.count()) / 1e9;
      double timeRemaining = timeElapsed * (allTests - testsDone) / testsDone;
      printTimeHumanReadable(timeRemaining);
      cout << " left)";
    }
    cout << "\r" << flush;
  }
}

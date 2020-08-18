#include "testCommons.h"

#include <chrono>
#include <cstdlib>
#include <iostream>

#include "color.h"
#include "commons.h"
#include "oddHoles.h"
#include "perfect.h"

using namespace std::chrono;
using std::cerr;
using std::cout;
using std::default_random_engine;
using std::flush;
using std::make_pair;
using std::make_tuple;
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

map<tuple<algos, bool, int>, double> sumTime;
// map<pair<int, bool>, double> sumClockTime;
map<tuple<algos, bool, int>, int> casesTested;
string algo_names[] = {"Perfect", "Naive", "CUDA Naive", "Cuda Perfect"};

// map<pair<int, bool>, double> sumTimeNaive;
// map<pair<int, bool>, double> sumClockTimeNaive;
// map<pair<int, bool>, int> casesTestedNaive;

default_random_engine generator;
normal_distribution<double> distribution(0.5, 0.15);

bool testWithStats(const Graph &G, algos algo, cuIsPerfectFunction cuFunction) {
  nanoseconds start_ns = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
  clock_t start_clock = clock();

  bool result;

  switch (algo) {
    case algoPerfect:
      result = isPerfectGraph(G);
      break;

    case algoNaive:
      result = isPerfectGraphNaive(G);
      break;

    case algoCudaNaive:
      assert(cuFunction != nullptr);
      result = cuFunction(G);
      break;

    case algoCudaPerfect:
      throw invalid_argument("Not implemented");

    default:
      throw invalid_argument("TestWithStats invalid argument");
  }

  nanoseconds end_ns = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
  clock_t end_clock = clock();

  double duration = (end_ns.count() - start_ns.count()) / 1e9;

  auto t = make_tuple(algo, result, G.n);
  sumTime[t] += duration;
  casesTested[t]++;

  return result;
}

void printStats() {
  for (int algo = algoPerfect; algo < algo_last; algo++) {
    int count = 0;
    for (auto it = sumTime.begin(); it != sumTime.end(); it++) {
      if (get<0>(it->first) == algo) count++;
    }

    if (count > 0) cout << algo_names[algo] << " recognition stats: " << endl;
    for (auto it = sumTime.begin(); it != sumTime.end(); it++) {
      if (get<0>(it->first) == algo) {
        int cases = casesTested[it->first];
        cout << "\tn=" << get<2>(it->first) << ", result=" << get<1>(it->first) << ", cases=" << cases
             << ", avgTime=" << it->second / cases << endl;
      }
    }
  }
}

bool StatsFactory::curStarted = false;
int StatsFactory::testCaseNr = 0;
int StatsFactory::curTestPartNr = 0;
algos StatsFactory::algo = algo_last;
int StatsFactory::curN = 0;
vec<string> StatsFactory::partNames = vec<string>();
vec<double> StatsFactory::curTime = vec<double>();
RaiiTimer StatsFactory::curTimer = RaiiTimer("");

map<tuple<algos, bool, int, int>, int> StatsFactory::mapCount = map<tuple<algos, bool, int, int>, int>();
map<tuple<algos, bool, int, int>, double> StatsFactory::mapSumTime =
    map<tuple<algos, bool, int, int>, double>();

void StatsFactory::startTestCase(const Graph &G, algos algo_in) {
  testCaseNr++;
  curStarted = true;
  if (testCaseNr == 1) {
    algo = algo_in;
  }
  curTestPartNr = 0;
  curN = G.n;
  curTimer = RaiiTimer("");
}

void StatsFactory::startTestCasePart(const string &name) {
  if (curTestPartNr > 0) {
    curTime.push_back(curTimer.getElapsedSeconds());
  }

  if (testCaseNr == 1) {
    partNames.push_back(name);
  } else {
    assert(partNames[curTestPartNr] == name);
  }
  curTestPartNr++;

  curTimer = RaiiTimer("");
}

void StatsFactory::endTestCase(bool result) {
  if (curTestPartNr > 0) {
    curTime.push_back(curTimer.getElapsedSeconds());
  }

  for (int i = 0; i < partNames.size(); i++) {
    auto t = make_tuple(algo, result, curN, i);
    mapCount[t]++;
    mapSumTime[t] += curTime[i];
  }
}

void StatsFactory::printStats2() {
  cout << "algorithm, result, n, num_runs, ";
  for (int i = 0; i < partNames.size(); i++) {
    cout << partNames[i];
    if (i + 1 < partNames.size()) cout << ", ";
  }
  cout << endl;

  for (auto it = mapCount.begin(); it != mapCount.end(); it++) {
    auto t = it->first;
    int count = it->second;
    cout << get<0>(t) << ", " << get<1>(t) << ", " << get<2>(t) << ", " << count << ", ";

    for (auto it2 = it; it != mapCount.end() && get<0>(it2->first) == get<0>(t) &&
                        get<1>(it2->first) == get<1>(t) && get<2>(it2->first) == get<2>(t);
         it2++) {
      assert(count == mapCount[it2->first]);
      cout << mapSumTime[it2->first] / count << ", ";
    }
    cout << endl;
  }
}

double getDistr() {
  double distr = distribution(generator);
  if (distr <= 0 || distr > 1) distr = 0.5;

  return distr;
}

void testGraph(const Graph &G, bool verbose) {
  if (verbose) cout << "Testing " << G.n << " vs naive" << endl;

  bool naivePerfect = testWithStats(G, algoNaive);

  bool perfect = testWithStats(G, algoPerfect);

  if (naivePerfect != perfect) {
    cout << "ERROR: " << endl << "naive=" << naivePerfect << endl << "perfect=" << perfect << endl;
    cout << G << endl;
    if (!naivePerfect) cout << findOddHoleNaive(G) << endl;
  }

  assert(naivePerfect == perfect);
}

void testGraph(const Graph &G, bool result, bool verbose) {
  if (verbose) cout << "Testing " << G.n << " vs " << result << endl;

  bool perfect = testWithStats(G, algoPerfect);

  if (perfect != result) {
    cout << "Error Test Graph" << endl;
    cout << G << endl;
    cout << "Expected " << result << ", got " << perfect << endl;
  }

  assert(perfect == result);
}

void printTimeHumanReadable(double time, bool use_cerr) {
  auto &out = use_cerr ? cerr : cout;

  double s = time;
  int h = s / (60 * 60);
  s -= h * (60 * 60);
  if (h != 0) {
    out << h << "h";
  }

  int m = s / 60;
  s -= m * 60;
  if (m != 0) {
    out << m << "m";
  }

  out << static_cast<int>(s) + 1 << "s";
}

RaiiProgressBar::RaiiProgressBar(int allTests, bool use_cerr) : allTests(allTests), use_cerr(use_cerr) {
  start_ns = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
  update(0);
}

RaiiProgressBar::~RaiiProgressBar() {
  if (use_cerr)
    cerr << endl;
  else
    cout << endl;
}

int RaiiProgressBar::getFilled(int testsDone) {
  double progress = testsDone / (static_cast<double>(allTests));
  return width * progress;
}

void RaiiProgressBar::update(int testsDone) {
  auto &out = use_cerr ? cerr : cout;

  int toFill = getFilled(testsDone);
  if (testsDone < 10 || testsDone == allTests || toFill != getFilled(testsDone - 1)) {
    out << "[";
    for (int i = 0; i < width; i++) {
      out << (i < toFill ? "X" : " ");
    }
    out << "]";
    if (testsDone > 0) {
      out << " (about ";
      nanoseconds end_ns = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
      double timeElapsed = (end_ns.count() - start_ns.count()) / 1e9;
      double timeRemaining = timeElapsed * (allTests - testsDone) / testsDone;
      printTimeHumanReadable(timeRemaining, use_cerr);
      out << " left)";
    }
    out << "\r" << flush;
  }
}

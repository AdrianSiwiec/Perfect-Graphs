#include "testCommons.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include "commons.h"
#include "perfect.h"
// Unit test of testCommons

using namespace std;

void testRandomGraphs() {
  auto graphs = getRandomGraphs(12, 1, 15);
  assert(graphs.size() == 15);
  for (int i = 0; i < 15; i++) {
    assert(graphs[i].n == 12);
    for (int v = 0; v < 12; v++) {
      for (int u = 0; u < 12; u++)
        if (u != v)
          assert(graphs[i].areNeighbours(u, v) == 1);
        else
          assert(graphs[i].areNeighbours(u, v) == 0);
    }
  }

  graphs = getRandomGraphs(15, 0.5, 15);
  for (int i = 0; i < 15; i++) {
    for (int v = 0; v < 15; v++) {
      for (int u = 0; u < 15; u++) {
        assert(graphs[i].areNeighbours(u, v) == graphs[i].areNeighbours(v, u));
      }
    }
  }
}

void testGetRandomPerfect() {
  for (int i = 0; i < 10; i++) {
    Graph G = getRandomPerfectGraph(6, 0.5);
    assert(isPerfectGraph(G));
  }
}

void testGetFullBinary() {
  Graph G = getFullBinaryTree(15);
  int leaves = 0;
  int edges = 0;
  for (int i = 0; i < 15; i++) {
    if (G[i].size() == 1) leaves++;
    edges += G[i].size();
  }

  edges /= 2;
  assert(edges == 14);
  assert(leaves == 8);
}

void testTuples() {
  assert(isAllZeros({0, 0, 0, 0}));
  assert(!isAllZeros({1, 0, 0, 0, 0, 0}));
  assert(!isAllZeros({0, 0, 0, 0, 0, 1}));

  assert(nextTuple({1, 0, 0, 1}, 3) == vec<int>({2, 0, 0, 1}));
  assert(nextTuple({2, 0, 0, 1}, 3) == vec<int>({0, 1, 0, 1}));

  assert(generateTuples(4, 3).size() == 3 * 3 * 3 * 3);
}

void testStatsFactory() {
  {
    RaiiProgressBar bar(4 * 5 * 2 * 2);

    for (int algo = algoPerfect; algo < algoCudaNaive; algo++) {
      for (int n = 5; n < 9; n++) {
        for (int i = 0; i < 5; i++) {
          for (int res = 0; res < 2; res++) {
            StatsFactory::startTestCase(getRandomGraph(n, 0), (algos)algo);

            StatsFactory::startTestCasePart("100ms (twice) + res/2");
            this_thread::sleep_for(chrono::milliseconds(100 + res * 50));

            StatsFactory::startTestCasePart("i + algo");
            this_thread::sleep_for(chrono::milliseconds((i + algo) * 100));

            if (((algo == algoPerfect) ^ ((n % 2) == 0)) && i != 0) {
              StatsFactory::startTestCasePart("50");
              this_thread::sleep_for(chrono::milliseconds(50));
            }

            if (i != 0) {
              StatsFactory::startTestCasePart("200");
              if (i % 2) {
                this_thread::sleep_for(chrono::milliseconds(200));
              } else {
                this_thread::sleep_for(chrono::milliseconds(50));
              }
            }

            StatsFactory::startTestCasePart("n-4");
            this_thread::sleep_for(chrono::milliseconds((n - 4) * 100));

            StatsFactory::startTestCasePart("100ms (twice) + res/2");
            this_thread::sleep_for(chrono::milliseconds(100));

            if (i != 0) {
              StatsFactory::startTestCasePart("200");
              if (i % 2) {
                this_thread::sleep_for(chrono::milliseconds(0));
              } else {
                this_thread::sleep_for(chrono::milliseconds(150));
              }
            }

            StatsFactory::endTestCase(res);

            bar.update(algo * 40 + (n - 5) * 10 + i * 2 + res);
          }
        }
      }
    }
  }

  StatsFactory::printStats2();
}

int main() {
  init();
  testRandomGraphs();
  testGetRandomPerfect();
  testTuples();
  testGetFullBinary();
  // testStatsFactory();
}

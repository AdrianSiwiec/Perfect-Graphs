#include "commons.h"
#include "oddHoles.h"
#include "cuCommons.h"
#include "perfect.h"
#include "testCommons.h"

#include <chrono>
#include <thread>

#include "src/devCode.dev"

void testPreparePathStart(context_t &context) {
  int *dev;
  CUCHECK(cudaMalloc(&dev, sizeof(int) * 5));

  transform(
      [=] MGPU_DEVICE(int id) {
        devPreparePathStart((7 * 11 + 3) * 11 + 5, dev, 3, 11);
        assert(dev[0] == 5);
        assert(dev[1] == 3);
        assert(dev[2] == 7);
        assert(dev[3] == 0);

        devPreparePathStart(((7 * 11 + 3) * 11 + 5) * 11 + 10, dev, 4, 11);
        assert(dev[0] == 10);
        assert(dev[1] == 5);
        assert(dev[2] == 3);
        assert(dev[3] == 7);
        assert(dev[4] == 0);

        devPreparePathStart(7 * 7 * 7 * 7 - 1, dev, 4, 7);
        assert(dev[0] == 6);
        assert(dev[1] == 6);
        assert(dev[2] == 6);
        assert(dev[3] == 6);
        assert(dev[4] == 0);

        devPreparePathStart(7, dev, 4, 7);
        assert(dev[0] == 0);
        assert(dev[1] == 1);
        assert(dev[2] == 0);
        assert(dev[3] == 0);
        assert(dev[4] == 0);
      },
      1, context);

  context.synchronize();
  CUCHECK(cudaFree(dev));
}

void testDevAreNeighbors(context_t &context) {
  Graph G = getRandomGraph(10, 0.5);
  CuGraph CG(G, context);

  int *ans = new int[100];
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      ans[i * 10 + j] = G.areNeighbours(i, j);
    }
  }

  int *devAns;
  CUCHECK(cudaMalloc(&devAns, sizeof(int) * 100));
  CUCHECK(cudaMemcpy(devAns, ans, sizeof(int) * 100, cudaMemcpyHostToDevice));

  free(ans);

  transform(
      [=] MGPU_DEVICE(int id) {
        for (int i = 0; i < 10; i++) {
          for (int j = 0; j < 10; j++) {
            assert(devAns[i * 10 + j] == devAreNeighbors(CG, i, j));
          }
        }
      },
      1, context);

  context.synchronize();

  CUCHECK(cudaFree(devAns));
}

void testDevIsDistinctValues(context_t &context) {
  transform(
      [=] MGPU_DEVICE(int id) {
        int dev[10];
        dev[0] = 0;
        dev[1] = 1;
        dev[2] = 2;
        dev[3] = 2;

        assert(devIsDistinctValues(dev, 3));
        assert(!devIsDistinctValues(dev, 4));
        assert(devIsDistinctValues(dev, 1));
        assert(devIsDistinctValues(dev, 0));
      },
      1, context);
}

void testDevIsAPath(context_t &context) {
  Graph G(6,
          "\
  .XX...\
  X.XX..\
  XX....\
  .X..X.\
  ...X.X\
  ....X.\
  ");

  CuGraph CG(G, context);

  transform(
      [=] MGPU_DEVICE(int id) {
        int a1[] = {0, 1};
        assert(devIsAPath(CG, a1, 2));
        int a2[] = {0, 2};
        assert(devIsAPath(CG, a2, 2));
        int a3[] = {0, 3};
        assert(!devIsAPath(CG, a3, 2));
        int a4[] = {0, 1, 3};
        assert(devIsAPath(CG, a4, 3));
        int a5[] = {0, 1, 2};
        assert(!devIsAPath(CG, a5, 3));
        int a6[] = {0, 1, 0};
        assert(!devIsAPath(CG, a6, 3));
        int a7[] = {0, 1, 2, 0};
        assert(!devIsAPath(CG, a7, 4));
        int a8[] = {0, 1, 3, 4, 5};
        assert(devIsAPath(CG, a8, 5));
        int a9[] = {0, 1, 3, 5};
        assert(!devIsAPath(CG, a9, 4));

        int a10[] = {0, 1, 2};
        assert(devIsAPath(CG, a10, 3, true));
        int a11[] = {2, 0, 1, 3};
        assert(!devIsAPath(CG, a11, 4, true, false));
        int a12[] = {2, 0, 1, 3};
        assert(devIsAPath(CG, a12, 4, true, true));
      },
      1, context);
  context.synchronize();

  G = Graph(6,
            "\
  .XX...\
  X.XX..\
  XX.X..\
  .XX.X.\
  ...X.X\
  ....X.\
  ");
  CuGraph CG2(G, context);

  transform(
      [=] MGPU_DEVICE(int id) {
        int a1[] = {2, 0, 1, 3};
        assert(devIsAPath(CG2, a1, 4, true, true));
        int a2[] = {0, 1, 2, 3};
        assert(devIsAPath(CG2, a2, 4, true, true));
      },
      1, context);
  context.synchronize();
}

void testDevNextNeighbor(context_t &context) {
  Graph G(7,
          "\
  ...X...\
  ..X.XX.\
  .X...X.\
  X......\
  .X.....\
  .XX....\
  .......\
  ");

  CuGraph CG(G, context);

  transform(
      [=] MGPU_DEVICE(int id) {
        assert(devGetFirstNeighbor(CG, 0) == 3);
        assert(devGetNextNeighbor(CG, 0, 3) == -1);
        assert(devGetFirstNeighbor(CG, 1) == 2);
        assert(devGetNextNeighbor(CG, 1, 2) == 4);
        assert(devGetNextNeighbor(CG, 1, 4) == 5);
        assert(devGetNextNeighbor(CG, 1, 5) == -1);
        assert(devGetFirstNeighbor(CG, 6) == -1);

        assert(CG.devNextNeighbor[0 * CG.n + 1] == -2);
      },
      1, context);
  context.synchronize();
}

void testDevNextPathInPlace(context_t &context) {
  Graph G(6,
          "\
  .XX...\
  X.XX..\
  XX....\
  .X..X.\
  ...X.X\
  ....X.\
  ");
  CuGraph CG(G, context);

  Graph G2(6,
           "\
  .XX...\
  X.XX..\
  XX.X..\
  .XX.X.\
  ...X.X\
  ....X.\
  ");
  CuGraph CG2(G2, context);

  transform(
      [=] MGPU_DEVICE(int id) {
        int v[5] = {100};
        int lenV = 0;

        devNextPathInPlace(CG, v, lenV, 2);
        assert(lenV == 2);
        assert(v[0] == 0);
        assert(v[1] == 1);

        devNextPathInPlace(CG, v, lenV, 2);
        assert(lenV == 2);
        assert(v[0] == 0);
        assert(v[1] == 2);

        devNextPathInPlace(CG, v, lenV, 2);
        assert(lenV == 2);
        assert(v[0] == 1);
        assert(v[1] == 0);

        devNextPathInPlace(CG, v, lenV, 2);
        assert(lenV == 2);
        assert(v[0] == 1);
        assert(v[1] == 2);

        devNextPathInPlace(CG, v, lenV, 2);
        assert(lenV == 2);
        assert(v[0] == 1);
        assert(v[1] == 3);

        devNextPathInPlace(CG, v, lenV, 2);
        assert(lenV == 2);
        assert(v[0] == 2);
        assert(v[1] == 0);

        v[0] = 5;
        v[1] = 4;
        lenV = 2;
        devNextPathInPlace(CG, v, lenV, 2);
        assert(lenV == 0);

        devNextPathInPlace(CG, v, lenV, 3);
        assert(lenV == 3);
        assert(v[0] == 0);
        assert(v[1] == 1);
        assert(v[2] == 3);

        devNextPathInPlace(CG, v, lenV, 3);
        assert(lenV == 3);
        assert(v[0] == 1);
        assert(v[1] == 3);
        assert(v[2] == 4);

        devNextPathInPlace(CG, v, lenV, 3);
        assert(lenV == 3);
        assert(v[0] == 2);
        assert(v[1] == 1);
        assert(v[2] == 3);

        lenV = 0;
        devNextPathInPlace(CG, v, lenV, 3, true);
        assert(lenV == 3);
        assert(v[0] == 0);
        assert(v[1] == 1);
        assert(v[2] == 2);

        devNextPathInPlace(CG, v, lenV, 3, true);
        assert(lenV == 3);
        assert(v[0] == 0);
        assert(v[1] == 1);
        assert(v[2] == 3);

        devNextPathInPlace(CG, v, lenV, 3, true);
        assert(lenV == 3);
        assert(v[0] == 0);
        assert(v[1] == 2);
        assert(v[2] == 1);

        lenV = 0;
        devNextPathInPlace(CG, v, lenV, 5);
        assert(lenV == 5);
        assert(v[0] == 0);
        assert(v[1] == 1);
        assert(v[2] == 3);
        assert(v[3] == 4);
        assert(v[4] == 5);

        lenV = 0;
        int counter = 0;
        do {
          devNextPathInPlace(CG, v, lenV, 4);
          assert(lenV == 4 || lenV == 0);
          counter++;
        } while (lenV != 0);
        assert(counter == 7);

        lenV = 0;
        counter = 0;
        do {
          devNextPathInPlace(CG, v, lenV, 3);
          assert(lenV == 3 || lenV == 0);
          counter++;
        } while (lenV != 0);
        assert(counter == 9);

        lenV = 0;
        counter = 0;
        do {
          devNextPathInPlace(CG, v, lenV, 3, true);
          assert(lenV == 3 || lenV == 0);
          counter++;
        } while (lenV != 0);
        assert(counter == 15);

        lenV = 0;
        devNextPathInPlace(CG2, v, lenV, 4, true, true);
        assert(lenV == 4);
        assert(v[0] == 0);
        assert(v[1] == 1);
        assert(v[2] == 2);
        assert(v[3] == 3);

        devNextPathInPlace(CG2, v, lenV, 4, true, true);
        assert(lenV == 4);
        assert(v[0] == 0);
        assert(v[1] == 1);
        assert(v[2] == 3);
        assert(v[3] == 2);

        devNextPathInPlace(CG2, v, lenV, 4, true, true);
        assert(lenV == 4);
        assert(v[0] == 0);
        assert(v[1] == 1);
        assert(v[2] == 3);
        assert(v[3] == 4);

        // Partial paths as an input
        v[0] = 1;
        v[1] = 3;
        lenV = 2;
        devNextPathInPlace(CG, v, lenV, 3);
        assert(lenV == 3);
        assert(v[0] == 1);
        assert(v[1] == 3);
        assert(v[2] == 4);

        v[0] = 3;
        lenV = 1;
        devNextPathInPlace(CG, v, lenV, 3);
        assert(lenV == 3);
        assert(v[0] == 3);
        assert(v[1] == 1);
        assert(v[2] == 0);
      },
      1, context);
  context.synchronize();
}

void testDevIsAHole(context_t &context) {
  Graph G(6,
          "\
  .X..X.\
  X.X...\
  .X.X..\
  ..X.X.\
  X..X..\
  ......\
  ");
  CuGraph CG(G, context);

  Graph G2(6,
           "\
  .X.XX.\
  X.X...\
  .X.X..\
  X.X.X.\
  X..X..\
  ......\
  ");
  CuGraph CG2(G2, context);

  transform(
      [=] MGPU_DEVICE(int id) {
        int a1[] = {0, 1, 2, 3, 4};
        assert(devIsAHole(CG, a1, 5));

        int a2[] = {3, 2, 1, 0, 4};
        assert(devIsAHole(CG, a2, 5));

        int a3[] = {0, 2, 3, 4};
        assert(!devIsAHole(CG, a3, 4));

        int a4[] = {1, 2, 3, 4};
        assert(!devIsAHole(CG, a4, 4));

        int a5[] = {0, 2, 1, 3, 4};
        assert(!devIsAHole(CG, a5, 5));

        int b1[] = {0, 1, 2, 3, 4};
        assert(!devIsAHole(CG2, b1, 5));

        int b2[] = {3, 2, 1, 0, 4};
        assert(!devIsAHole(CG2, b2, 5));

        int b3[] = {0, 1, 2, 3};
        assert(devIsAHole(CG2, b3, 4));
        assert(!devIsAHole(CG2, b3, 5));

        int b4[] = {1, 2, 3, 4};
        assert(!devIsAHole(CG2, b4, 4));

        int b5[] = {0, 2, 1, 3, 4};
        assert(!devIsAHole(CG2, b5, 5));
      },
      1, context);
  context.synchronize();
}

void testCuContainsHoleOfSize(context_t &context) {
  Graph G(6,
          "\
  .XX...\
  X.XX..\
  XX....\
  .X..X.\
  ...X.X\
  ....X.\
  ");
  CuGraph CG(G, context);

  assert(!cuContainsHoleOfSize(CG, 3, context));
  assert(!cuContainsHoleOfSize(CG, 4, context));
  assert(!cuContainsHoleOfSize(CG, 5, context));
  assert(!cuContainsHoleOfSize(CG, 6, context));

  G = Graph(6,
            "\
  .XX...\
  X.XX..\
  XX..X.\
  .X..X.\
  ..XX.X\
  ....X.\
  ");
  CuGraph CG2(G, context);
  assert(cuContainsHoleOfSize(CG2, 4, context));
  assert(!cuContainsHoleOfSize(CG2, 3, context));
  assert(!cuContainsHoleOfSize(CG2, 5, context));
  assert(!cuContainsHoleOfSize(CG2, 6, context));

  // assert(!cuContainsHoleOfSize(CG, 3, context, 6));
  // assert(!cuContainsHoleOfSize(CG, 4, context, 6));
  // assert(!cuContainsHoleOfSize(CG, 5, context, 6*6));
  // assert(!cuContainsHoleOfSize(CG, 6, context, 6*6*6));

  // assert(cuContainsHoleOfSize(CG2, 4, context, 6));
  // assert(!cuContainsHoleOfSize(CG2, 3, context, 6*6));
  // assert(!cuContainsHoleOfSize(CG2, 5, context, 6*6*6));
  // assert(!cuContainsHoleOfSize(CG2, 6, context, 6*6*6));
}

void testCuContainsOddHoleNaive(context_t &context) {
  Graph G(6,
          "\
  .XX...\
  X.XX..\
  XX...X\
  .X..X.\
  ...X.X\
  ..X.X.\
  ");
  CuGraph CG(G, context);

  assert(cuContainsOddHoleNaive(CG, context));

  G = Graph(6,
            "\
  .XX...\
  X.XX..\
  XX..XX\
  .X..X.\
  ..XX.X\
  ..X.X.\
  ");
  CuGraph CG2(G, context);
  assert(!cuContainsOddHoleNaive(CG2, context));

  G = Graph(5,
            "\
  .X..X\
  X.X..\
  .X.X.\
  ..X.X\
  X..X.\
  ");
  CuGraph CG3(G, context);
  assert(cuContainsOddHoleNaive(CG3, context));
}

map<pair<int, bool>, double> sumTimeDev;
map<pair<int, bool>, int> casesTestedDev;

map<pair<int, bool>, double> sumTimeNaiveDev;
map<pair<int, bool>, int> casesTestedNaiveDev;

map<pair<int, bool>, double> cuSumTimeNaive;
map<pair<int, bool>, int> cuCasesTestedNaive;

void printCuTestStats() {
  if (!sumTimeDev.empty()) cout << "Perfect recognition stats: " << endl;
  for (auto it = sumTimeDev.begin(); it != sumTimeDev.end(); it++) {
    if (!it->first.second) continue;
    int cases = casesTestedDev[it->first];
    cout << "\tn=" << it->first.first << ", result=" << it->first.second << ", cases=" << cases
         << ", avgTime=" << it->second / cases << endl;
  }
  for (auto it = sumTimeDev.begin(); it != sumTimeDev.end(); it++) {
    if (it->first.second) continue;
    int cases = casesTestedDev[it->first];
    cout << "\tn=" << it->first.first << ", result=" << it->first.second << ", cases=" << cases
         << ", avgTime=" << it->second / cases << endl;
  }

  if (!sumTimeNaiveDev.empty()) cout << "Naive recognition stats: " << endl;
  for (auto it = sumTimeNaiveDev.begin(); it != sumTimeNaiveDev.end(); it++) {
    if (!it->first.second) continue;
    int cases = casesTestedNaiveDev[it->first];
    cout << "\tn=" << it->first.first << ", result=" << it->first.second << ", cases=" << cases
         << ", avgTime=" << it->second / cases << endl;
  }
  for (auto it = sumTimeNaiveDev.begin(); it != sumTimeNaiveDev.end(); it++) {
    if (it->first.second) continue;
    int cases = casesTestedNaiveDev[it->first];
    cout << "\tn=" << it->first.first << ", result=" << it->first.second << ", cases=" << cases
         << ", avgTime=" << it->second / cases << endl;
  }

  if (!cuSumTimeNaive.empty()) cout << "CUDA recognition stats: " << endl;
  for (auto it = cuSumTimeNaive.begin(); it != cuSumTimeNaive.end(); it++) {
    if (!it->first.second) continue;
    int cases = cuCasesTestedNaive[it->first];
    cout << "\tn=" << it->first.first << ", result=" << it->first.second << ", cases=" << cases
         << ", avgTime=" << it->second / cases << endl;
  }
  for (auto it = cuSumTimeNaive.begin(); it != cuSumTimeNaive.end(); it++) {
    if (it->first.second) continue;
    int cases = cuCasesTestedNaive[it->first];
    cout << "\tn=" << it->first.first << ", result=" << it->first.second << ", cases=" << cases
         << ", avgTime=" << it->second / cases << endl;
  }

  cout << endl;
}

void cuTestGraphSimpleWithStats(const Graph &G, context_t &context, bool runNaive = true) {
  string N = to_string(G.n);
  bool res;

  {
    RaiiTimer t("");
    res = isPerfectGraph(G);
    pair<int, bool> p = make_pair(G.n, res);

    sumTimeDev[p] += t.getElapsedSeconds();

    casesTestedDev[p]++;
  }

  if (runNaive) {
    RaiiTimer t("");
    if (isPerfectGraphNaive(G) != res) {
      cout << "Error NAIVE" << endl << G << endl;
      exit(1);
    }

    double elapsed = t.getElapsedSeconds();

    auto p = make_pair(G.n, res);
    sumTimeNaiveDev[p] += elapsed;
    casesTestedNaiveDev[p]++;
  }

  {
    RaiiTimer t("");
    int cuRes;
    if ((cuRes = cuIsPerfectNaive(G, context, 100000000)) != res) {
      cout << "Error CUDA NAIVE" << endl << G << endl;
      cout << "res: " << res << ", cuRes: " << cuRes << endl;
      exit(1);
    }
    double elapsed = t.getElapsedSeconds();
    auto p = make_pair(G.n, res);
    cuSumTimeNaive[p] += elapsed;
    cuCasesTestedNaive[p]++;
  }

  printCuTestStats();
}
void testCuIsPerfectNaiveHand(context_t &context) {
  Graph G(11,
          "\
  .XXXXX.....\
  X.XX..X....\
  XX.X.....X.\
  XXX.....X.X\
  X....X....X\
  X...X.X..X.\
  .X...X...X.\
  ........X..\
  ...X...X..X\
  ..X..XX....\
  ...XX...X..\
  ");

  cuTestGraphSimpleWithStats(G, context, false);

  G = Graph(10,
            "\
  .XXXX.....\
  X.XX..XX..\
  XX.X.X...X\
  XXX.......\
  X.....X...\
  ..X......X\
  .X..X..X..\
  .X....X.XX\
  .......X.X\
  ..X..X.XX.\
  ");
  cuTestGraphSimpleWithStats(G, context, false);

  G = Graph(11,
            "\
    .X...XXX...\
    X.........X\
    ...X..X....\
    ..X.X.X....\
    ...X...X.XX\
    X.....XXX..\
    X.XX.X.X...\
    X...XXX..XX\
    .....X...X.\
    ....X..XX.X\
    .X..X..X.X.\
    ");
  cuTestGraphSimpleWithStats(G, context, false);

  G = Graph(10,
            "\
    .X..X..XX.\
    X..XXXXXX.\
    .....X.X..\
    .X....XX..\
    XX....X..X\
    .XX....XX.\
    .X.XX...XX\
    XXXX.X...X\
    XX...XX...\
    ....X.XX..\
    ");
  cout << findOddHoleNaive(G) << endl;
}

void testCuIsPerfectNaive(context_t &context) {
  // for (int i = 5; i < 16; i++) {
  //   Graph G = getRandomGraph(i, 0.5);
  //   cuTestGraphSimple(G, context);
  // }

  for (int i = 0; i < 20; i++) {
    cuTestGraphSimpleWithStats(getRandomGraph(10, 0.5), context);
  }

  for (int i = 0; i < 20; i++) {
    cuTestGraphSimpleWithStats(getRandomGraph(11, 0.5), context, false);
  }

  for (int i = 0; i < 10; i++) {
    cuTestGraphSimpleWithStats(getBipariteGraph(8, 0.5).getLineGraph(), context);
  }

  // for (int i = 0; i < 30; i++) {
  //   cuTestGraphSimpleWithStats(getBipariteGraph(9, 0.5).getLineGraph(), context, false);
  // }
}

int main() {
  init();
  standard_context_t context(0);

  testPreparePathStart(context);
  testDevAreNeighbors(context);
  testDevIsDistinctValues(context);
  testDevIsAPath(context);
  testDevNextNeighbor(context);
  testDevNextPathInPlace(context);
  testDevIsAHole(context);
  testCuContainsHoleOfSize(context);
  testCuContainsOddHoleNaive(context);
  testCuIsPerfectNaiveHand(context);
  testCuIsPerfectNaive(context);

  context.synchronize();
}
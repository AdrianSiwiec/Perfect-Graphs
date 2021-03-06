#include "commons.h"
#include "cuCommons.h"
#include "oddHoles.h"
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
            assert(devAns[i * 10 + j] == devAreNeighbors(CG.devMatrix, CG.n, i, j));
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
        assert(devIsAPath(CG.devMatrix, CG.n, a1, 2));
        int a2[] = {0, 2};
        assert(devIsAPath(CG.devMatrix, CG.n, a2, 2));
        int a3[] = {0, 3};
        assert(!devIsAPath(CG.devMatrix, CG.n, a3, 2));
        int a4[] = {0, 1, 3};
        assert(devIsAPath(CG.devMatrix, CG.n, a4, 3));
        int a5[] = {0, 1, 2};
        assert(!devIsAPath(CG.devMatrix, CG.n, a5, 3));
        int a6[] = {0, 1, 0};
        assert(!devIsAPath(CG.devMatrix, CG.n, a6, 3));
        int a7[] = {0, 1, 2, 0};
        assert(!devIsAPath(CG.devMatrix, CG.n, a7, 4));
        int a8[] = {0, 1, 3, 4, 5};
        assert(devIsAPath(CG.devMatrix, CG.n, a8, 5));
        int a9[] = {0, 1, 3, 5};
        assert(!devIsAPath(CG.devMatrix, CG.n, a9, 4));

        int a10[] = {0, 1, 2};
        assert(devIsAPath(CG.devMatrix, CG.n, a10, 3, true));
        int a11[] = {2, 0, 1, 3};
        assert(!devIsAPath(CG.devMatrix, CG.n, a11, 4, true, false));
        int a12[] = {2, 0, 1, 3};
        assert(devIsAPath(CG.devMatrix, CG.n, a12, 4, true, true));
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
        assert(devIsAPath(CG2.devMatrix, CG2.n, a1, 4, true, true));
        int a2[] = {0, 1, 2, 3};
        assert(devIsAPath(CG2.devMatrix, CG2.n, a2, 4, true, true));
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
        assert(devGetFirstNeighbor(CG.devFirstNeighbor, 0) == 3);
        assert(devGetNextNeighbor(CG.devNextNeighbor, CG.n, 0, 3) == -1);
        assert(devGetFirstNeighbor(CG.devFirstNeighbor, 1) == 2);
        assert(devGetNextNeighbor(CG.devNextNeighbor, CG.n, 1, 2) == 4);
        assert(devGetNextNeighbor(CG.devNextNeighbor, CG.n, 1, 4) == 5);
        assert(devGetNextNeighbor(CG.devNextNeighbor, CG.n, 1, 5) == -1);
        assert(devGetFirstNeighbor(CG.devFirstNeighbor, 6) == -1);

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

        devNextPathInPlace(CG.devMatrix, CG.devNextNeighbor, CG.devFirstNeighbor, CG.n, v, lenV, 2);
        assert(lenV == 2);
        assert(v[0] == 0);
        assert(v[1] == 1);

        devNextPathInPlace(CG.devMatrix, CG.devNextNeighbor, CG.devFirstNeighbor, CG.n, v, lenV, 2);
        assert(lenV == 2);
        assert(v[0] == 0);
        assert(v[1] == 2);

        devNextPathInPlace(CG.devMatrix, CG.devNextNeighbor, CG.devFirstNeighbor, CG.n, v, lenV, 2);
        assert(lenV == 2);
        assert(v[0] == 1);
        assert(v[1] == 0);

        devNextPathInPlace(CG.devMatrix, CG.devNextNeighbor, CG.devFirstNeighbor, CG.n, v, lenV, 2);
        assert(lenV == 2);
        assert(v[0] == 1);
        assert(v[1] == 2);

        devNextPathInPlace(CG.devMatrix, CG.devNextNeighbor, CG.devFirstNeighbor, CG.n, v, lenV, 2);
        assert(lenV == 2);
        assert(v[0] == 1);
        assert(v[1] == 3);

        devNextPathInPlace(CG.devMatrix, CG.devNextNeighbor, CG.devFirstNeighbor, CG.n, v, lenV, 2);
        assert(lenV == 2);
        assert(v[0] == 2);
        assert(v[1] == 0);

        v[0] = 5;
        v[1] = 4;
        lenV = 2;
        devNextPathInPlace(CG.devMatrix, CG.devNextNeighbor, CG.devFirstNeighbor, CG.n, v, lenV, 2);
        assert(lenV == 0);

        devNextPathInPlace(CG.devMatrix, CG.devNextNeighbor, CG.devFirstNeighbor, CG.n, v, lenV, 3);
        assert(lenV == 3);
        assert(v[0] == 0);
        assert(v[1] == 1);
        assert(v[2] == 3);

        devNextPathInPlace(CG.devMatrix, CG.devNextNeighbor, CG.devFirstNeighbor, CG.n, v, lenV, 3);
        assert(lenV == 3);
        assert(v[0] == 1);
        assert(v[1] == 3);
        assert(v[2] == 4);

        devNextPathInPlace(CG.devMatrix, CG.devNextNeighbor, CG.devFirstNeighbor, CG.n, v, lenV, 3);
        assert(lenV == 3);
        assert(v[0] == 2);
        assert(v[1] == 1);
        assert(v[2] == 3);

        lenV = 0;
        devNextPathInPlace(CG.devMatrix, CG.devNextNeighbor, CG.devFirstNeighbor, CG.n, v, lenV, 3, true);
        assert(lenV == 3);
        assert(v[0] == 0);
        assert(v[1] == 1);
        assert(v[2] == 2);

        devNextPathInPlace(CG.devMatrix, CG.devNextNeighbor, CG.devFirstNeighbor, CG.n, v, lenV, 3, true);
        assert(lenV == 3);
        assert(v[0] == 0);
        assert(v[1] == 1);
        assert(v[2] == 3);

        devNextPathInPlace(CG.devMatrix, CG.devNextNeighbor, CG.devFirstNeighbor, CG.n, v, lenV, 3, true);
        assert(lenV == 3);
        assert(v[0] == 0);
        assert(v[1] == 2);
        assert(v[2] == 1);

        lenV = 0;
        devNextPathInPlace(CG.devMatrix, CG.devNextNeighbor, CG.devFirstNeighbor, CG.n, v, lenV, 5);
        assert(lenV == 5);
        assert(v[0] == 0);
        assert(v[1] == 1);
        assert(v[2] == 3);
        assert(v[3] == 4);
        assert(v[4] == 5);

        lenV = 0;
        int counter = 0;
        do {
          devNextPathInPlace(CG.devMatrix, CG.devNextNeighbor, CG.devFirstNeighbor, CG.n, v, lenV, 4);
          assert(lenV == 4 || lenV == 0);
          counter++;
        } while (lenV != 0);
        assert(counter == 7);

        lenV = 0;
        counter = 0;
        do {
          devNextPathInPlace(CG.devMatrix, CG.devNextNeighbor, CG.devFirstNeighbor, CG.n, v, lenV, 3);
          assert(lenV == 3 || lenV == 0);
          counter++;
        } while (lenV != 0);
        assert(counter == 9);

        lenV = 0;
        counter = 0;
        do {
          devNextPathInPlace(CG.devMatrix, CG.devNextNeighbor, CG.devFirstNeighbor, CG.n, v, lenV, 3, true);
          assert(lenV == 3 || lenV == 0);
          counter++;
        } while (lenV != 0);
        assert(counter == 15);

        lenV = 0;
        devNextPathInPlace(CG2.devMatrix, CG2.devNextNeighbor, CG2.devFirstNeighbor, CG2.n, v, lenV, 4, true,
                           true);
        assert(lenV == 4);
        assert(v[0] == 0);
        assert(v[1] == 1);
        assert(v[2] == 2);
        assert(v[3] == 3);

        devNextPathInPlace(CG2.devMatrix, CG2.devNextNeighbor, CG2.devFirstNeighbor, CG2.n, v, lenV, 4, true,
                           true);
        assert(lenV == 4);
        assert(v[0] == 0);
        assert(v[1] == 1);
        assert(v[2] == 3);
        assert(v[3] == 2);

        devNextPathInPlace(CG2.devMatrix, CG2.devNextNeighbor, CG2.devFirstNeighbor, CG2.n, v, lenV, 4, true,
                           true);
        assert(lenV == 4);
        assert(v[0] == 0);
        assert(v[1] == 1);
        assert(v[2] == 3);
        assert(v[3] == 4);

        // Partial paths as an input
        v[0] = 1;
        v[1] = 3;
        lenV = 2;
        devNextPathInPlace(CG.devMatrix, CG.devNextNeighbor, CG.devFirstNeighbor, CG.n, v, lenV, 3);
        assert(lenV == 3);
        assert(v[0] == 1);
        assert(v[1] == 3);
        assert(v[2] == 4);

        v[0] = 3;
        lenV = 1;
        devNextPathInPlace(CG.devMatrix, CG.devNextNeighbor, CG.devFirstNeighbor, CG.n, v, lenV, 3);
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
        assert(devIsAHole(CG.devMatrix, CG.n, a1, 5));

        int a2[] = {3, 2, 1, 0, 4};
        assert(devIsAHole(CG.devMatrix, CG.n, a2, 5));

        int a3[] = {0, 2, 3, 4};
        assert(!devIsAHole(CG.devMatrix, CG.n, a3, 4));

        int a4[] = {1, 2, 3, 4};
        assert(!devIsAHole(CG.devMatrix, CG.n, a4, 4));

        int a5[] = {0, 2, 1, 3, 4};
        assert(!devIsAHole(CG.devMatrix, CG.n, a5, 5));

        int b1[] = {0, 1, 2, 3, 4};
        assert(!devIsAHole(CG2.devMatrix, CG.n, b1, 5));

        int b2[] = {3, 2, 1, 0, 4};
        assert(!devIsAHole(CG2.devMatrix, CG.n, b2, 5));

        int b3[] = {0, 1, 2, 3};
        assert(devIsAHole(CG2.devMatrix, CG2.n, b3, 4));
        assert(!devIsAHole(CG2.devMatrix, CG2.n, b3, 5));

        int b4[] = {1, 2, 3, 4};
        assert(!devIsAHole(CG2.devMatrix, CG2.n, b4, 4));

        int b5[] = {0, 2, 1, 3, 4};
        assert(!devIsAHole(CG2.devMatrix, CG2.n, b5, 5));
      },
      1, context);
  context.synchronize();
}

// void testCuContainsHoleOfSize(context_t &context) {
//   Graph G(6,
//           "\
//   .XX...\
//   X.XX..\
//   XX....\
//   .X..X.\
//   ...X.X\
//   ....X.\
//   ");
//   CuGraph CG(G, context);

//   assert(!cuContainsHoleOfSize(CG, 3, context));
//   assert(!cuContainsHoleOfSize(CG, 4, context));
//   assert(!cuContainsHoleOfSize(CG, 5, context));
//   assert(!cuContainsHoleOfSize(CG, 6, context));

//   G = Graph(6,
//             "\
//   .XX...\
//   X.XX..\
//   XX..X.\
//   .X..X.\
//   ..XX.X\
//   ....X.\
//   ");
//   CuGraph CG2(G, context);
//   assert(cuContainsHoleOfSize(CG2, 4, context));
//   assert(!cuContainsHoleOfSize(CG2, 3, context));
//   assert(!cuContainsHoleOfSize(CG2, 5, context));
//   assert(!cuContainsHoleOfSize(CG2, 6, context));

//   // assert(!cuContainsHoleOfSize(CG, 3, context, 6));
//   // assert(!cuContainsHoleOfSize(CG, 4, context, 6));
//   // assert(!cuContainsHoleOfSize(CG, 5, context, 6*6));
//   // assert(!cuContainsHoleOfSize(CG, 6, context, 6*6*6));

//   // assert(cuContainsHoleOfSize(CG2, 4, context, 6));
//   // assert(!cuContainsHoleOfSize(CG2, 3, context, 6*6));
//   // assert(!cuContainsHoleOfSize(CG2, 5, context, 6*6*6));
//   // assert(!cuContainsHoleOfSize(CG2, 6, context, 6*6*6));
// }

// void testCuContainsOddHoleNaive(context_t &context) {
//   Graph G(6,
//           "\
//   .XX...\
//   X.XX..\
//   XX...X\
//   .X..X.\
//   ...X.X\
//   ..X.X.\
//   ");
//   CuGraph CG(G, context);

//   assert(cuContainsOddHoleNaive(CG, context));

//   G = Graph(6,
//             "\
//   .XX...\
//   X.XX..\
//   XX..XX\
//   .X..X.\
//   ..XX.X\
//   ..X.X.\
//   ");
//   CuGraph CG2(G, context);
//   assert(!cuContainsOddHoleNaive(CG2, context));

//   G = Graph(5,
//             "\
//   .X..X\
//   X.X..\
//   .X.X.\
//   ..X.X\
//   X..X.\
//   ");
//   CuGraph CG3(G, context);
//   assert(cuContainsOddHoleNaive(CG3, context));
// }

bool cuTestGraphSimple(const Graph &G, context_t &context, bool runNaive = true) {
  string N = to_string(G.n);
  bool res;

  res = isPerfectGraph(G);

  int cuRes;
  if (runNaive) {
    if ((cuRes = cuIsPerfectNaive(G)) != res) {
      cout << "Error CUDA NAIVE" << endl << G << endl;
      cout << "res: " << res << ", cuRes: " << cuRes << endl;
      exit(1);
    }
  }

  cuRes = cuIsPerfect(G);
  if (cuRes != res) {
    cout << "Error CUDA PERFECT" << endl << G << endl;
    cout << "res: " << res << ", cuRes: " << cuRes << endl;
    exit(1);
  }

  return res;
}
void testCuIsPerfectNaiveHand(context_t &context) {
  cuTestGraphSimple(Graph(8,
                          "\
.XXXX...\
X..XX..X\
X...X.XX\
XX...XXX\
XXX..XX.\
...XX..X\
..XXX..X\
.XXX.XX.\
"),
                    context, true);

  cuTestGraphSimple(Graph(8,
                          "\
..X..X.X\
......XX\
X...XXX.\
....XX..\
..XX..X.\
X.XX....\
.XX.X..X\
XX....X.\
"),
                    context, true);

  cuTestGraphSimple(Graph(8,
                          "\
.XXXXX..\
X..XX.XX\
X..X.X.X\
XXX.X..X\
XX.X.X..\
X.X.X.XX\
.X...X.X\
.XXX.XX.\
"),
                    context, true);

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

  cuTestGraphSimple(G, context, true);

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
  cuTestGraphSimple(G, context, true);

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
  cuTestGraphSimple(G, context, true);
}

void testCuIsPerfectNaive(context_t &context) {
  // for (int i = 5; i < 16; i++) {
  //   Graph G = getRandomGraph(i, 0.5);
  //   cuTestGraphSimple(G, context);
  // }

  // RaiiProgressBar bar(20 + 20 + 10 + 100 + 100 + 100);

  for (int i = 0; i < 20; i++) {
    cuTestGraphSimple(getRandomGraph(10, 0.5), context, true);
    // bar.update(i);
  }

  for (int i = 0; i < 20; i++) {
    cuTestGraphSimple(getRandomGraph(11, 0.5), context, true);
    // bar.update(i + 20);
  }

  for (int i = 0; i < 10; i++) {
    Graph G = getBipariteGraph(8, 0.5).getLineGraph();

    cuTestGraphSimple(G, context, true);

    // bar.update(i + 40);
  }

  // for (int i = 0; i < 100; i++) {
  //   cuTestGraphSimple(getBipariteGraph(9, 0.5).getLineGraph(), context, false);
  //   bar.update(i + 50);
  // }

  // for (int i = 0; i < 100; i++) {
  //   cuTestGraphSimple(getBipariteGraph(10, 0.5).getLineGraph(), context, false);
  //   bar.update(i + 150);
  // }

  // for (int i = 0; i < 10; i++) {
  //   cuTestGraphSimple(getBipariteGraph(11, 0.5).getLineGraph(), context, false);
  //   bar.update(i*10 + 250);
  // }

  // printCuTestStats();
}

void cutestPerfectVsNaive(context_t &context) {
  for (int i = 0; i < (bigTests ? 30 : 100); i++) {
    Graph G = getRandomGraph(bigTests ? 8 : 6, 0.5);
    cuTestGraphSimple(G, context, true);
  }
}

void cutestNonPerfect(context_t &context) {
  for (int i = 0; i < 100; i++) {
    Graph G = getNonPerfectGraph(9, 10, 0.5);
    cuTestGraphSimple(G, context, true);
  }
}

void cutestBiparite(context_t &context) {
  for (int i = 0; i < (bigTests ? 100 : 20); i++) {
    Graph G = getBipariteGraph(bigTests ? 10 : 7, 0.5);
    cuTestGraphSimple(G, context, true);
  }
}

void cutestLineBiparite(context_t &context) {
  for (int i = 0; i < (bigTests ? 30 : 20); i++) {
    Graph G = getBipariteGraph(bigTests ? 9 : 7, 0.5).getLineGraph();

    cuTestGraphSimple(G, context, true);
  }
}

void cuTestPerfectHandInteresting(context_t &context) {
  cuTestGraphSimple(Graph(15,
                          "\
...X..........X\
...X.X...X...X.\
.......XX.XX...\
XX..X..........\
...X.X...X...X.\
.X..X...X......\
..........X..X.\
..X........XXX.\
..X..X....XX.X.\
.X..X........X.\
..X...X.X..X.X.\
..X....XX.X..X.\
.......X......X\
.X..X.XXXXXX...\
X...........X..\
"),
                    context, true);

  cuTestGraphSimple(Graph(17,
                          "\
.........X.......\
..X..X....XX....X\
.X.XXXX.X..X..XX.\
..X.X...X...X.X.X\
..XX..XX..XXXXX.X\
.XX........X..X.X\
..X.X..XX.XXX.X.X\
....X.X.X.....XX.\
..XX..XX.X.XXXX..\
X.......X...X....\
.X..X.X....X..XX.\
.XX.XXX.X.X.X.XXX\
...XX.X.XX.X..X..\
....X...X........\
..XXXXXXX.XXX..XX\
..X....X..XX..X..\
.X.XXXX....X..X..\
"),
                    context, true);

  cuTestGraphSimple(Graph(10,
                          "\
.XX..XXXXX\
X.XXX..XXX\
XX...X.XXX\
.X..XXX.XX\
.X.X.XXX.X\
X.XXX.X.X.\
X..XXX.XXX\
XXX.X.X.XX\
XXXX.XXX..\
XXXXX.XX..\
"),
                    context, true);
}

int main() {
  init();
  standard_context_t context(0);

  testPreparePathStart(context);
  testDevAreNeighbors(context);
  testDevIsDistinctValues(context);
  testDevIsAPath(context);
  testDevNextNeighbor(context);
  // testDevNextPathInPlace(context);
  testDevIsAHole(context);
  // testCuContainsHoleOfSize(context);
  // testCuContainsOddHoleNaive(context);
  testCuIsPerfectNaiveHand(context);
  cuTestPerfectHandInteresting(context);
  testCuIsPerfectNaive(context);
  cutestPerfectVsNaive(context);
  cutestNonPerfect(context);
  cutestBiparite(context);
  cutestLineBiparite(context);

  context.synchronize();
}
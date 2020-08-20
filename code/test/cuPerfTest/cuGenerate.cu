#include <ctime>
#include <map>
#include <random>
#include "commons.h"
#include "oddHoles.h"
#include "perfect.h"
#include "testCommons.h"

#include "src/devCode.dev"

int main() {
  srand(time(0));

  for (int k = 0; k < 5; k++) {
    double p = getDistr();
    Graph G(0);
    int counter = 0;
    do {
      G = getRandomGraph(20, p);
      StatsFactory::startTestCase(G, algoCudaPerfect);
      counter++;
      if (counter % 100 == 0) {
        StatsFactory::printStats2();
      }
      StatsFactory::endTestCase(false);
    } while (!cuIsPerfect(G, true));

    cout << G << endl;
  }
}
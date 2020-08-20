#include <ctime>
#include <map>
#include <random>
#include "commons.h"
#include "oddHoles.h"
#include "perfect.h"
#include "testCommons.h"

int main() {
  srand(time(0));

  for (int k = 0; k < 5; k++) {
    double p = getDistr();
    Graph G(0);
    int counter = 0;
    do {
      G = getRandomGraph(20, p);
      counter++;
      if (counter % 100000 == 0) {
        p = getDistr();
      }
    } while (!isPerfectGraph(G, false));

    cout << G << endl;
  }
}
#include <ctime>
#include <map>
#include <random>
#include "commons.h"
#include "oddHoles.h"
#include "perfect.h"
#include "testCommons.h"

int main() {
  srand(time(0));

  int howManyToGenerate = 10;
  int counter = 0;

  while (true) {
    Graph G(0);
    do {
      G = getBipariteGraph(15, getDistr()).getLineGraph();
    } while (G.n != 26);

    cout << G << endl;
    counter++;
    if (counter == howManyToGenerate) return 0;
  }
}
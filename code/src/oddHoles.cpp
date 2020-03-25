#include "oddHoles.h"
#include "commons.h"

bool isT1(const Graph &G, const vec<int> &v) {
  if (v.size() != 5)
    return false;

  if (!isDistinctValues(v))
    return false;

  for (int i : v)
    if (i < 0 || i >= G.n)
      return false;

  for (int i = 0; i < 5; i++) {
    for (int j = i + 1; j < 5; j++) {
      if (abs(i - j) == 1 || abs(i - j) == 4) {
        if (!G.areNeighbours(v[i], v[j]))
          return false;
      } else {
        if (G.areNeighbours(v[i], v[j]))
          return false;
      }
    }
  }

  return true;
}
vec<int> findT1(const Graph &G) {
  vec<int> v(5);
  do {
    if (isT1(G, v))
      return v;

    nextTupleInPlace(v, G.n);
  } while (!isAllZeros(v));

  return vec<int>();
}
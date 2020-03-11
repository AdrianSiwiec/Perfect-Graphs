#include "pyramids.h"
#include "commons.h"

bool checkPrerequisites(const Graph &G, const vec<int> &b, const int a,
                        const vec<int> &s) {

  // We assume b is a Triangle and s is an EmptyStarTriangle

  int aAdjB = 0;

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      if (i == j)
        continue;
      if (b[i] == s[j]) {
        return false;
      }
      if ((G[b[i]][s[j]] && b[j] != s[j]) || G[s[i]][s[j]]) {
        return false;
      }
    }
  }

  for (int i = 0; i < 2; i++) {
    if (!G[a][s[i]])
      return false;
      
    if (G[a][b[i]]) {
      aAdjB++;
      if (aAdjB > 1)
        return false;
      if (b[i] != s[i])
        return false;
    }
  }

  return true;
}

bool containsPyramide(const Graph &G) {
  auto t = findPyramide(G);
  return get<0>(t).size() > 0;
}

tuple<vec<int>, int, vec<vec<int>>> findPyramide(const Graph &G) {
  auto triangles = getTriangles(G);
  auto emptyStars = getEmptyStarTriangles(G);

  for (auto triangle : triangles) {
    for (auto eStar : emptyStars) {
      const vec<int> &b = triangle;
      const int a = eStar.st;
      const vec<int> &s = eStar.nd;

      if (!checkPrerequisites(G, b, a, s))
        continue;
    }
  }
}
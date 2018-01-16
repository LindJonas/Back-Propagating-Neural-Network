#include <iostream>
#include <cmath>
#include <cstdlib>

using namespace std;

int main()
{
  cout << "2 5 1" << endl;
  for(int i = 0; i < 2000; i++)
  {
    int n1 = (int)(2.0 * rand() / double(RAND_MAX));
    int n2 = (int)(2.0 * rand() / double(RAND_MAX));
    int t = n1 ^ n2; // 0 or 1
    cout << n1 << ' ' << n2 << endl;
    cout << t << endl;
  }

  return 0;
}

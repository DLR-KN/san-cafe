#include <complex>
#include <exception>
#include <iostream>
#include <vector>

struct cafe_exception : std::exception {
  const char* what() const noexcept { return "Out of coffe!\n"; }
};

int main()
{
  std::cout << "Starting Test of SAN-CAFE" << std::endl;
  throw cafe_exception();

  return 0;
}

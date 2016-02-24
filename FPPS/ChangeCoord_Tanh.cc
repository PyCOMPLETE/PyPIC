#include "ChangeCoord_Tanh.h"
#include "FunctionsFPPS.h"
#include <cmath>

  ChangeCoord_Tanh::ChangeCoord_Tanh(const double& a_):
  ChangeCoord(),
  a(a_)
  {}
ChangeCoord_Tanh::~ChangeCoord_Tanh() {}

double ChangeCoord_Tanh::f(const double& x) const 
{
  return atanh(x)/a;
}
double ChangeCoord_Tanh::inv_f(const double& x) const
{
  return tanh(a*x);
}

double ChangeCoord_Tanh::der_f(const double& x) const
{
  return 1./(1.-a*sq2(x))/a;
}
double ChangeCoord_Tanh::der_der_f(const double& x) const
{
  return 2.*x/sq2((1.-sq2(x)))/a;
}

double ChangeCoord_Tanh::getA() const
{
  return a;
}

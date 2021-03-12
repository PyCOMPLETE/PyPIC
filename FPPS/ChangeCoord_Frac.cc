#include "ChangeCoord_Frac.h"
#include "FunctionsFPPS.h"
#include <cmath>

ChangeCoord_Frac::ChangeCoord_Frac(const double& a_):
ChangeCoord(),a(a_)
{}
ChangeCoord_Frac::~ChangeCoord_Frac() {}

double ChangeCoord_Frac::f(const double& x) const 
{
  return a*x/(1.0-fabs(x));
}
double ChangeCoord_Frac::inv_f(const double& x) const
{
  return x/(fabs(x)+a);
}

double ChangeCoord_Frac::der_f(const double& x) const
{
  return a/sq2(1.0-fabs(x));
}
double ChangeCoord_Frac::der_der_f(const double& x) const
{
  return 2.0*a/sq2(1.0-fabs(x))/(1.0-fabs(x))*x/fabs(x);
}

double ChangeCoord_Frac::getA() const
{
  return a;
}

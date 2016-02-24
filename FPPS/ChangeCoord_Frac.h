#ifndef COORDCHANGEFRAC
#define COORDCHANGEFRAC

#include "ChangeCoord.h"

class ChangeCoord_Frac: public ChangeCoord{

public:

  ChangeCoord_Frac(const double& a_);
  virtual ~ChangeCoord_Frac();

  double f(const double& x) const;
  double inv_f(const double& x) const;

  double der_f(const double& x) const;
  double der_der_f(const double& x) const;

  double getA() const;

private:

  const double a;

};

#endif

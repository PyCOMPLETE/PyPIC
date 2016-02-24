#ifndef COORDCHANGETANH
#define COORDCHANGETANH

#include "ChangeCoord.h"

class ChangeCoord_Tanh: public ChangeCoord{

public:

  ChangeCoord_Tanh(const double& a_);
  virtual ~ChangeCoord_Tanh();

  double f(const double& x) const;
  double inv_f(const double& x) const;

  double der_f(const double& x) const;
  double der_der_f(const double& x) const;

  double getA() const;

private:

  const double a;

};

#endif

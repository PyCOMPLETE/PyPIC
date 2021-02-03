#ifndef COORDCHANGE
#define COORDCHANGE

class ChangeCoord{

public:

  ChangeCoord();
  virtual ~ChangeCoord();

  virtual double f(const double& x) const=0;
  virtual double inv_f(const double& x) const=0;

  virtual double der_f(const double& x) const=0;
  virtual double der_der_f(const double& x) const=0;

  virtual double getA() const=0;

};

#endif

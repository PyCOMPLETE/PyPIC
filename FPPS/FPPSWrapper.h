#ifndef __FPPSTEST
#define __FPPSTEST


#include "NonLinearMesh.h"
#include "ChangeCoord_Frac.h"
#include "ChargeDistribution.h"
#include "FastPolarPoissonSolver.h"
#include "ElectricFieldSolver.h"
#include "PolarBeamRepresentation.h"

class FPPSWrapper{
public:
    FPPSWrapper();
    virtual ~FPPSWrapper();
    void scatter(double* x,double* y,double* charge,int n);
    void gather(double* x,double* y,double* Ex, double* Ey,int n);
	void solve();
    void useSourceAsProbe();

protected:
    bool sourceIsProbe;
	ChangeCoord_Frac *g;
	Mesh *chargeDistributionMesh;
	Mesh *radialField;
	Mesh *polarField;
	ChargeDistribution *chargeDistribution;
	FastPolarPoissonSolver *fastPolarPoissonSolver;
	ElectricFieldSolver *electricFieldSolver;
    PolarBeamRepresentation* sourcePolarBeamRepresentation;
    PolarBeamRepresentation* probePolarBeamRepresentation;
};

class FPPSUniform : public FPPSWrapper {
public:
	FPPSUniform(int nTheta, int nR, double r);
};

class FPPSOpenBoundary : public FPPSWrapper {
public:
	FPPSOpenBoundary(int nTheta, int nR, double a);
    virtual ~FPPSOpenBoundary();
protected:
	ChangeCoord_Frac *g;
};

#endif

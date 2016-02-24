#ifndef ELECTRICFIELD_SOLVER
#define ELECTRICFIELD_SOLVER

#include "Mesh.h"
#include "FastPolarPoissonSolver.h"
#include "PolarBeamRepresentation.h"

class ElectricFieldSolver {
public:
    ElectricFieldSolver(FastPolarPoissonSolver* s,Mesh* rf,Mesh* pf);
    virtual ~ElectricFieldSolver();
    
    void solve();
    void getField(PolarBeamRepresentation* polarBeamRepresentation,double* x,double* y,double* Ex,double* Ey) const;

protected:
    FastPolarPoissonSolver* solver;
    Mesh* radialField;
    Mesh* polarField;
};



#endif

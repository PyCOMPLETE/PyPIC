#ifndef FASTPOLARPOISSONSOLVER
#define FASTPOLARPOISSONSOLVER

#include <complex.h>
#include <fftw3.h>
#include <vector>
#include "Mesh.h"
#include "ChargeDistribution.h"


class FastPolarPoissonSolver{
    public:
    FastPolarPoissonSolver(ChargeDistribution* c);
    virtual ~FastPolarPoissonSolver();

    virtual void solve();

    Mesh* getPotential();

    protected:
    ChargeDistribution* chargeDistribution;
	fftw_complex** fourier;
    fftw_complex* fourierData;
	fftw_plan* plan_direct;
	fftw_plan* plan_inverse;

    void ThomasAlgorithm(const std::vector<double>& a, std::vector<double>& b, const std::vector<double>& c, fftw_complex** d, const int& n) const;
    virtual std::vector<double> constructLowerDiag() const;
    virtual void constructMainDiag(const int& n, std::vector<double>& v) const;
    virtual std::vector<double> constructUpperDiag() const;
    static double const EPSILON0 = 8.854187817e-12;

    void printFourierToFile(std::string fileNameReal,std::string fileNameImag);

};

#endif

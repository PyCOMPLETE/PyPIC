
#include "FastPolarPoissonSolver.h"
#include <iostream>
#include <sstream>
#include <fstream>

FastPolarPoissonSolver::FastPolarPoissonSolver(ChargeDistribution* c):
chargeDistribution(c)
{
    fourier = new fftw_complex* [chargeDistribution->getMesh()->getRadialSize()];
	fourierData = new fftw_complex [chargeDistribution->getMesh()->getRadialSize()*chargeDistribution->getMesh()->getPolarSize()];
    plan_direct = new fftw_plan [chargeDistribution->getMesh()->getRadialSize()];
    plan_inverse = new fftw_plan [chargeDistribution->getMesh()->getRadialSize()];
	for(int i=0; i<chargeDistribution->getMesh()->getRadialSize();++i){
		fourier[i]= &(fourierData[i*chargeDistribution->getMesh()->getPolarSize()]);
		plan_direct[i] = fftw_plan_dft_r2c_1d(chargeDistribution->getMesh()->getPolarSize(),chargeDistribution->getMesh()->getRingValues(i),fourier[i],FFTW_MEASURE);
		plan_inverse[i] = fftw_plan_dft_c2r_1d(chargeDistribution->getMesh()->getPolarSize(),fourier[i],chargeDistribution->getMesh()->getRingValues(i), FFTW_MEASURE);
	}
}

FastPolarPoissonSolver::~FastPolarPoissonSolver(){
    delete fourierData;
    delete fourier;
    delete plan_direct;
    delete plan_inverse;
};

Mesh* FastPolarPoissonSolver::getPotential() {
    return chargeDistribution->getMesh();
}

void FastPolarPoissonSolver::FastPolarPoissonSolver::solve()
{

#ifdef TIMING
  timeval Start;
  timeval End;
#endif

#ifdef TIMING
    gettimeofday(&Start,NULL);
#endif

#pragma omp parallel for schedule(guided,10)
    for(int i=0; i<chargeDistribution->getMesh()->getRadialSize(); ++i){ //@fixed r, compute fft u(r,theta)
	    fftw_execute(plan_direct[i]);
    }
//    std::stringstream fileNameStreamReal;
//    fileNameStreamReal << "FFTReal.csv";
//    std::string fileNameReal = fileNameStreamReal.str();
//    std::stringstream fileNameStreamImag;
//    fileNameStreamImag << "FFTImag.csv";
//    std::string fileNameImag = fileNameStreamImag.str();
//    printFourierToFile(fileNameReal,fileNameImag);    

#ifdef TIMING
    
    gettimeofday(&End,NULL);
	double Duration = End.tv_sec-Start.tv_sec+(End.tv_usec-Start.tv_usec)/1E6;
	std::cout<<"Time for fftw execute "<<Duration<<" s"<<std::endl;
#endif
    //TODO make this members of the class allocated at instanciation with a reset function
	const std::vector<double> lowerDiag = constructLowerDiag();
	const std::vector<double> upperDiag = constructUpperDiag();
	std::vector<double> mainDiag(chargeDistribution->getMesh()->getRadialSize()+1);
	int nt=chargeDistribution->getMesh()->getPolarSize()/2+1;

#ifdef TIMING
    gettimeofday(&Start,NULL);
#endif

#pragma omp parallel for firstprivate(mainDiag) schedule(guided,100)
	for(int i=0; i<nt;++i){ //@fixed n, compute un(r)
		
		constructMainDiag(i,mainDiag);
		ThomasAlgorithm(lowerDiag,mainDiag,upperDiag,fourier,i);

	}

//    std::stringstream fileNameStreamReal2;
//    fileNameStreamReal2 << "FFTReal_a.csv";
//    std::string fileNameReal2 = fileNameStreamReal2.str();
//    std::stringstream fileNameStreamImag2;
//    fileNameStreamImag2 << "FFTImag_a.csv";
//    std::string fileNameImag2 = fileNameStreamImag2.str();
//    printFourierToFile(fileNameReal2,fileNameImag2);    

#ifdef TIMING
    gettimeofday(&End,NULL);
    Duration = End.tv_sec-Start.tv_sec+(End.tv_usec-Start.tv_usec)/1E6;
	std::cout<<"Time for thomas algorithm "<<Duration<<" s"<<std::endl;
    gettimeofday(&Start,NULL);
#endif

#pragma omp parallel for schedule(guided,100)
	for (int i = 0; i < chargeDistribution->getMesh()->getRadialSize(); ++i) //@fixed r, compute fftback(un(r))
	{
		fftw_execute(plan_inverse[i]);
	}
#ifdef TIMING
    gettimeofday(&End,NULL);
    Duration = End.tv_sec-Start.tv_sec+(End.tv_usec-Start.tv_usec)/1E6;
	std::cout<<"Time for fftw execute inv "<<Duration<<" s"<<std::endl;
#endif

}

void FastPolarPoissonSolver::ThomasAlgorithm(const std::vector<double>& a, std::vector<double>& b, const std::vector<double>& c, fftw_complex** d, const int& n) const
{
  // Solve the tridiagonal linear system Ax=D , with A a tridiagonal matrice given by its lower diagonal a, main diagonal b, upper diagonal c.
// algorithm from http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)

	int dim= b.size();


	if(((int)a.size())!=dim-1){
		std::cerr<< "Problem in the Thomas Algorithm of the Poisson Solver.\nThe lower diagonal a size does not match with the density given in input.\nDim a= " << a.size() <<" whereas dim d= " << dim<< std::endl;
	} 
	else if(((int)c.size())!=dim-1){
		std::cerr<< "Problem in the Thomas Algorithm of the Poisson Solver.\nThe upper diagonal c size does not match with the density given in input.\nDim c= " << c.size() <<" whereas dim d= " << dim<< std::endl;
	} 
	else if(((int)b.size())!=dim){
		std::cerr<< "Problem in the Thomas Algorithm of the Poisson Solver.\nThe main diagonal b size does not match with the density given in input.\nDim b= " << b.size() <<" whereas dim d= " << dim<< std::endl;
	} 
	else{
		//std::vector<double>  tmpb(b);
		//std::vector<double>  tmpd(d);

		double m =0.0;
		double complex BC=0.0;
		
		for(int j=1; j<dim; ++j)
		{
			if(j<dim-1){
				d[j][n]/=(-EPSILON0*chargeDistribution->getMesh()->getPolarSize());
				m=a[j-1]/b[j-1];
				b[j]=b[j]-m*c[j-1];
				d[j][n]=d[j][n]-(m)*d[j-1][n];
			}
			else
			{
				m=a[j-1]/b[j-1];
				b[j]=b[j]-m*c[j-1];
				BC=BC-(m)*d[j-1][n];				
			}
		}

		BC=BC/(b[dim-1]);
		d[dim-2][n]=(d[dim-2][n]-(c[dim-2])*BC)/(b[dim-2]);
		for(int j=dim-3; j>=0; --j)
		{
			d[j][n]=(d[j][n]-(c[j])*d[j+1][n])/(b[j]);
		}

	}

}


//Construction of the linear system

std::vector<double> FastPolarPoissonSolver::constructLowerDiag() const
{
	std::vector<double> v(chargeDistribution->getMesh()->getRadialSize());
	for(int i=0;i<(int)v.size();++i)
	{
	  v[i] = chargeDistribution->getMesh()->getLaplaceLower(i);
	}
	return v;
}

void FastPolarPoissonSolver::constructMainDiag(const int& n, std::vector<double>& v) const
{
	for(int i=0;i<((int)v.size());++i)
	{
	  v[i] = chargeDistribution->getMesh()->getLaplaceDiag(n,i);
	}
}
std::vector<double> FastPolarPoissonSolver::constructUpperDiag() const
{
	std::vector<double> v(chargeDistribution->getMesh()->getRadialSize());
	for(int i=0;i<(int)v.size();++i)
	{
	  v[i] = chargeDistribution->getMesh()->getLaplaceUpper(i);
    }
	return v;
}

void FastPolarPoissonSolver::printFourierToFile(std::string fileNameReal,std::string fileNameImag) {

    std::ofstream* outReal;
    outReal=new std::ofstream;
    outReal->open(fileNameReal.c_str());
    std::ofstream* outImag;
    outImag=new std::ofstream;
    outImag->open(fileNameImag.c_str());

    for(int i=0; i<chargeDistribution->getMesh()->getRadialSize(); ++i){
        for(int j=0;j<chargeDistribution->getMesh()->getPolarSize();++j){
            *outReal << creal(fourier[i][j]);
            *outImag << cimag(fourier[i][j]);
            if(j<chargeDistribution->getMesh()->getPolarSize()-1){
                *outReal << ",";
                *outImag << ",";
            }
        }
        *outReal << std::endl;
        *outImag << std::endl;
    }
    outReal->close();
    outImag->close();
}

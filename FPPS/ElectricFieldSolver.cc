
#include "ElectricFieldSolver.h"
#include "FunctionsFPPS.h"
#include <sstream>
#include <cmath>
#include <sys/time.h>

ElectricFieldSolver::ElectricFieldSolver(FastPolarPoissonSolver* s,Mesh* rf,Mesh* pf):
solver(s),radialField(rf),polarField(pf)
{}

ElectricFieldSolver::~ElectricFieldSolver(){}

void ElectricFieldSolver::solve() {

    solver->solve();
//    std::stringstream fileNameStream;
//    fileNameStream << "Potential_"<< beam <<".csv";
//    std::string fileName = fileNameStream.str();
//    solver->getPotential()->writeToFile(fileName);
#ifdef TIMING
    timeval Start;
    timeval End;
    gettimeofday(&Start,NULL);
#endif
#pragma omp parallel for default(none) collapse(2) schedule(guided,100)
	for(int i=0; i<radialField->getRadialSize(); ++i)
	{	
		for (int j=0; j<radialField->getPolarSize(); ++j)
		{
            //std::cout<<"computing radial field "<<i<<" "<<j<<std::endl;
			radialField->setValue(i,j,-solver->getPotential()->getGradR(i,j));
            //std::cout<<"computing polar field "<<i<<" "<<j<<std::endl;
			polarField->setValue(i,j,-solver->getPotential()->getGradTheta(i,j));
            //std::cout<<"done "<<i<<" "<<j<<std::endl;
		}
	}

//    std::stringstream fileNameStream2;
//    fileNameStream2 << "RadialField_"<< beam <<".csv";
//    fileName = fileNameStream2.str();
//    radialField->writeToFile(fileName);
//    std::stringstream fileNameStream3;
//    fileNameStream3 << "PolarField_"<< beam <<".csv";
//    fileName = fileNameStream3.str();
//    polarField->writeToFile(fileName);

#ifdef TIMING
    gettimeofday(&End,NULL);
	double Duration = End.tv_sec-Start.tv_sec+(End.tv_usec-Start.tv_usec)/1E6;
    std::cout<<"Time for field computation "<<Duration<<" s"<<std::endl;
#endif
}

void ElectricFieldSolver::getField(PolarBeamRepresentation* polarBeamRepresentation,double* x,double* y,double* Ex,double* Ey) const
{
    double r(0),Fr(0),Ftheta(0),r_i(0),r_iplus1(0),theta_i(0),norm(0),aTheta(0),aRadius(0),bRadius(0);
    int r_ind(0),theta_ind(0),theta_next(0),theta_opp(0),theta_opp_next(0);

    #pragma omp parallel for shared(polarBeamRepresentation) private(r,r_i,r_iplus1,theta_i,norm,aTheta,aRadius,bRadius,Fr,Ftheta,r_ind,theta_ind,theta_next,theta_opp,theta_opp_next) schedule(guided,1000)
    for(int i=0; i<polarBeamRepresentation->getSize(); ++i)
    {
        r=radialField->computeR(polarBeamRepresentation->getRadius(i));
        r_ind=radialField->which_R(r);
        theta_ind=radialField->which_Theta(polarBeamRepresentation->getAngle(i));
        theta_i=radialField->getTheta(theta_ind);
        aTheta=(polarBeamRepresentation->getAngle(i)-theta_i)/radialField->getDeltaTheta();
        theta_next = theta_ind+1;
        if(theta_next == radialField->getPolarSize()) theta_next = 0;

        if(r_ind==radialField->getRadialSize()-1){
            Fr=radialField->getValue(r_ind,theta_ind)*(1.0-aTheta)+radialField->getValue(r_ind,theta_next)*aTheta;
            Ftheta=polarField->getValue(r_ind,theta_ind)*(1.0-aTheta)+polarField->getValue(r_ind,theta_next)*aTheta;
        } else if (r_ind == -1){
            r_i=radialField->getR(0);
            theta_opp = theta_ind+radialField->getPolarSize()/2;
            if(theta_opp>radialField->getPolarSize()) theta_opp -= radialField->getPolarSize();
            theta_opp_next = theta_opp-1;
            if(theta_opp_next == -1) theta_opp_next = radialField->getPolarSize()-1;

            norm=2.0*sq2(r_i);
            aRadius = (sq2(r_i)-sq2(r))/norm;
            bRadius = (sq2(r_i)+sq2(r))/norm;

            Fr=(aTheta*aRadius)*radialField->getValue(0,theta_opp_next)
                +((1.0-aTheta)*aRadius)*radialField->getValue(0,theta_opp)
                +(aTheta*bRadius)*radialField->getValue(0,theta_next)
                +((1.0-aTheta)*bRadius)*radialField->getValue(0,theta_ind);

            Ftheta=(aTheta*aRadius)*polarField->getValue(0,theta_opp_next)
                +((1.0-aTheta)*aRadius)*polarField->getValue(0,theta_opp)
                +(aTheta*bRadius)*polarField->getValue(0,theta_next)
                +((1.0-aTheta)*bRadius)*polarField->getValue(0,theta_ind);
        } else if(r_ind < radialField->getRadialSize()-1){
            r_i=radialField->getR(r_ind);
            r_iplus1=r_i+radialField->getDeltaR();

            norm=(sq2(r_iplus1)-sq2(r_i));
            aRadius = (sq2(r)-sq2(r_i))/norm;
            bRadius = (sq2(r_iplus1)-sq2(r))/norm;

            Fr=(aTheta*aRadius)*radialField->getValue(r_ind+1,theta_next)
                +((1.0-aTheta)*aRadius)*radialField->getValue(r_ind+1,theta_ind)
                +(aTheta*bRadius)*radialField->getValue(r_ind,theta_next)
                +((1.0-aTheta)*bRadius)*radialField->getValue(r_ind,theta_ind);

            Ftheta=(aTheta*aRadius)*polarField->getValue(r_ind+1,theta_next)
                +((1.0-aTheta)*aRadius)*polarField->getValue(r_ind+1,theta_ind)
                +(aTheta*bRadius)*polarField->getValue(r_ind,theta_next)
                +((1.0-aTheta)*bRadius)*polarField->getValue(r_ind,theta_ind);
        } else{
            continue;         //Ignoring particles outside of the grid
        }
        polarBeamRepresentation->getField(i,x,y,Ex,Ey,Fr,Ftheta);
    }
}

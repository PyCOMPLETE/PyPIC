#include "ChargeDistribution.h"
#include "FunctionsFPPS.h"
#include <iostream>

ChargeDistribution::ChargeDistribution(Mesh* m):
  mesh(m)
{}

ChargeDistribution::~ChargeDistribution(){};

Mesh* ChargeDistribution::getMesh() {
    return mesh;
}

void ChargeDistribution::fill(PolarBeamRepresentation* polarBeamRepresentation){
    mesh->reset();
    #pragma omp parallel for shared(polarBeamRepresentation) schedule(guided,1000)
    for(int i=0; i<polarBeamRepresentation->getSize(); ++i){
	    addParticle(polarBeamRepresentation->getRadius(i),polarBeamRepresentation->getAngle(i),polarBeamRepresentation->getCharge(i));
	}
    mesh->normalise();
}

void ChargeDistribution::addParticle(const double& rReal, const double& theta, const double& charge){
  double r = mesh->computeR(rReal);
  int r_ind=mesh->which_R(r);
  int theta_ind=mesh->which_Theta(theta);
  int theta_next = theta_ind + 1;
  if(theta_next==mesh->getPolarSize()) theta_next = 0;
  double theta_i=mesh->getTheta(theta_ind);
  double aTheta=(theta-theta_i)/mesh->getDeltaTheta();

  if(r_ind==-1){
    double theta_opp = theta_ind+mesh->getPolarSize()/2;
    if(theta_opp>mesh->getPolarSize()) theta_opp -= mesh->getPolarSize();
    double theta_opp_next = theta_opp-1;
    if(theta_opp_next == -1) theta_opp_next = mesh->getPolarSize()-1;
    double r_i=mesh->getR(0);
    double norm = 2.0*sq2(r_i);
    double aRadius = (sq2(r_i)+sq2(r))/norm;
    double bRadius = (sq2(r_i)-sq2(r))/norm;
    mesh->getValue(0,theta_ind)+=(1.0-aTheta)*bRadius*charge;
    mesh->getValue(0,theta_next)+=aTheta*bRadius*charge;
    mesh->getValue(0,theta_opp)+=(1.0-aTheta)*aRadius*charge;
    mesh->getValue(0,theta_opp_next)+=aTheta*aRadius*charge;
  }else if(r_ind<mesh->getRadialSize()-1){
    double r_i=mesh->getR(r_ind);
    double r_iplus1=r_i+mesh->getDeltaR();
    double norm = sq2(r_iplus1)-sq2(r_i);  // normalisation factor for the itnerpolation + charge
    double aRadius = (sq2(r)-sq2(r_i))/norm;
    double bRadius = (sq2(r_iplus1)-sq2(r))/norm;
    mesh->getValue(r_ind,theta_ind)+=(1.0-aTheta)*bRadius*charge;
    mesh->getValue(r_ind,theta_next)+=aTheta*bRadius*charge;
    mesh->getValue(r_ind+1,theta_ind)+=(1.0-aTheta)*aRadius*charge;
    mesh->getValue(r_ind+1,theta_next)+=aTheta*aRadius*charge;
  }
  //Ignoring particles outside of the grid
  //else {
  //mesh->getValue(mesh->getRadialSize()-1,theta_ind)+=charge*aTheta;
  //mesh->getValue(mesh->getRadialSize()-1,theta_next)+=charge*(1-aTheta);
  //}
}

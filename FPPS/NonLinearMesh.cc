#include "NonLinearMesh.h"
#include "FunctionsFPPS.h"
#include <cmath>
#include <iostream>

NonLinearMesh::NonLinearMesh(const int& n_theta, const int& m_r, const double& r_max,ChangeCoord* c):
Mesh(n_theta,m_r,r_max),changeCoord(c)
{}

NonLinearMesh::~NonLinearMesh(){};

double NonLinearMesh::computeR(const double& r) const {
    return changeCoord->inv_f(Mesh::computeR(r));
}

double NonLinearMesh::getLaplaceUpper(const int& i) const {
    double der = changeCoord->der_f(getR(i));
    return 1./(sq2(der)*sq2(getDeltaR()))+(1./(changeCoord->f(getR(i))*der)-changeCoord->der_der_f(getR(i))/(sq2(der)*der))/(2.0*getDeltaR());
}
double NonLinearMesh::getLaplaceDiag(const int& n,const int& i) const {
    if(i==getRadialSize()) return 1.0;
    else if(i==0) {
        double x = getR(0);
        double f = changeCoord->f(x);
        double der = changeCoord->der_f(x);
        return -2./(sq2(der)*sq2(getDeltaR()))-(double)(n*n)/sq2(f)+pow(-1.0,n)*(1.0/(sq2(der)*sq2(getDeltaR()))-(1./(f*der)-changeCoord->der_der_f(x)/(sq2(der)*der))/(2.0*getDeltaR()));
    }else return -2./(sq2(changeCoord->der_f(getR(i)))*sq2(getDeltaR()))-(double)(n*n)/sq2(changeCoord->f(getR(i)));
}
double NonLinearMesh::getLaplaceLower(const int& i) const{
    if(i==getRadialSize()-1) return 0.0;
    else {
        double der = changeCoord->der_f(getR(i+1));
        return 1./(sq2(der)*sq2(getDeltaR()))-(1./(changeCoord->f(getR(i+1))*der)-changeCoord->der_der_f(getR(i+1))/(sq2(der)*der))/(2.0*getDeltaR());
    }
}

double NonLinearMesh::getGradR(const int& i, const int& j) const {
    return Mesh::getGradR(i,j)/changeCoord->der_f(getR(i));
}

double NonLinearMesh::getGradTheta(const int& i, const int& j) const {
    int jPrev = j-1;
    if(jPrev == -1) jPrev = getPolarSize()-1;
    int jNext = j+1;
    if(jNext == getPolarSize()) jNext = 0;
    return (data[i][jNext]-data[i][jPrev])/(2.0*getDeltaTheta()*changeCoord->f(getR(i)));
}


// normalise to the area of each cell
void NonLinearMesh::normalise(){
	for(int i=0;i<getRadialSize();++i){
		for(int j=0; j<getPolarSize(); ++j){
		   data[i][j]/=getDeltaR()*changeCoord->f(getR(i))*getDeltaTheta()*changeCoord->der_f(getR(i));
		}
	}
}

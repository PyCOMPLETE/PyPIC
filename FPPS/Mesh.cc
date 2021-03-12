#include "Mesh.h"
#include "FunctionsFPPS.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

//Constructors


Mesh::Mesh(const int& n_j, const int& m_r, const double& r_max):
polarSize(n_j),
radialSize(m_r),
maxRadius(r_max){
    data=new double* [radialSize];
    internalData=new double [radialSize*polarSize];
    for(int i=0;i<radialSize;i++) data[i]= &(internalData[i*polarSize]);
}

//Destructor

Mesh::~Mesh(){
    delete internalData;
    delete data;
}


//Get


int Mesh::getPolarSize() const{
	return polarSize;
}

int Mesh::getRadialSize() const{
	return radialSize;
}

double Mesh::getMaxRadius() const{
	return maxRadius;
}

int Mesh::getSize() const{
	return radialSize*polarSize;
}

double Mesh::getDeltaR() const{
	return maxRadius/radialSize;
}
double Mesh::getDeltaTheta() const{
	return (2.0*M_PI/(polarSize));
}

double Mesh::getR(const int& i) const{
	if(i>radialSize or i<-1){
		std::cout<<"From Mesh::getR(int i): i should be 0<=i<="<<radialSize<<". i="<<i<<" ."<<std::endl;
		return -1;
	} else if(i==-1){
        return 0.0;
    }else return ((double)i+0.5)*getDeltaR();
}

double Mesh::getTheta(const int& i) const{
	if(i>polarSize or i<0){
		std::cerr<<"From Mesh::getTheta(int i): i should be 0<=i<="<<polarSize<<". i="<<i<<" ."<<std::endl;
		return -1;
	}
	else return (double)i*getDeltaTheta();
}

//Which


int Mesh::which_Theta(const double& j) const
{
	if(j<0 or j>2.0*M_PI){
		std::cerr<<"From Mesh::which_Theta(double j): j has to be in [0.0, 2*pi]. j=" << j << std::endl;
		return -1;
	}
	else return (int)floor(j/getDeltaTheta());
}
int Mesh::which_R(const double& r) const
{
	if(r>maxRadius){ 
		return maxRadius;
	}else if(r<0){
		std::cout<<"Error in Mesh::which_R(double R): R is negative"<<std::endl;
		return maxRadius;
    }else if(r<getDeltaR()/2){
        return -1;
    }else return (int)floor(r/getDeltaR()-0.5);
}

double Mesh::computeR(const double& r) const{
	return r;
}


double Mesh::computeTheta(const double& x, const double& y) const{
    double retVal = atan2(y,x);
    if (retVal < 0) retVal += 2.0*M_PI;
    return retVal;
}
/*
double Mesh::computeTheta(const double& x, const double& y) const{
	if(x>0){
		if(y>0)return atan(fabs(y/x));
		else if(y==0.0) return 0.0;
		else if(y<0) return 2.0*M_PI-atan(fabs(y/x));
		else{
			std::cerr<<"Problem in Mesh::computeTheta: x>0, y weird. 0.0 returned."<< std::endl;
			return 0.0;
		}
	}else if(x==0.0){
		if(y>0) return M_PI/2.0;
		else if(y==0.0){
			std::cerr<<"Problem in Mesh::computeTheta: x=y=0, j not defined. 0.0 returned."<< x << " " <<y<<std::endl;
			return 0.0;
		}else if(y<0) return 3.0*M_PI/2.0;
		else{
		  std::cerr<<"Problem in Mesh::computeTheta: x=0, y weird. 0.0 returned."<<x<< std::endl;
			return 0.0;
		}
	}else if(x<0){
		if(y>0) return M_PI-atan(fabs(y/x));
		else if(y==0.0) return M_PI;
		else if(y<0) return atan(fabs(y/x))+M_PI;
		else{
			std::cerr<<"Problem in Mesh::computeTheta: x<0, y weird. 0.0 returned."<< std::endl;
			return 0.0;
		} 
	}else{
		std::cerr<<"Problem in Mesh::computeTheta: x weird. 0.0 returned."<<x<<  std::endl;
		return 0.0;
	}
}
*/

void Mesh::setValue(const int& i, const int& j, double value){
  if(i<radialSize && j<polarSize) data[i][j]=value;
	else{
		std::cerr << " Problem with the input's dimension in Mesh::fill"<< std::endl;
	}
}

double Mesh::getLaplaceUpper(const int& i) const {
    return 1.0/(getDeltaR()*getDeltaR())+1.0/(2.0*getR(i)*getDeltaR());
}
double Mesh::getLaplaceDiag(const int& n,const int& i) const {
    if(i==getRadialSize()) return 1.0;
	else if (i==0) return -2.0/(getDeltaR()*getDeltaR())-(double)(n*n)/(double)(getR(0)*getR(0))+pow(-1.0,n)*(1.0/(getDeltaR()*getDeltaR())-1.0/(2.0*getR(0)*getDeltaR()));
    else return -2.0/(getDeltaR()*getDeltaR())-(double)(n*n)/(double)(getR(i)*getR(i));
}

double Mesh::getLaplaceLower(const int& i) const{
    if(i==getRadialSize()-1) return 0.0;
    else return 1.0/(getDeltaR()*getDeltaR())-1.0/(2.0*getR(i+1)*getDeltaR());
}

double Mesh::getGradR(const int& i, const int& j) const {
	if(i>0 && i<getRadialSize()-1) return (data[i+1][j]-data[i-1][j])/(2.0*getDeltaR());
	else if(i==0) return (data[1][j]-data[0][j])/getDeltaR();
	else if(i==getRadialSize()-1) return (data[i][j]-data[i-1][j])/getDeltaR();
	else{
		std::cerr<< "From computeEr: Problem with the input's dimension."<<std::endl;
		return 0.0;
	}
}

double Mesh::getGradTheta(const int& i, const int& j) const {
    int jPrev = j-1;
    if(jPrev == -1) jPrev = getPolarSize()-1;
    int jNext = j+1;
    if(jNext == getPolarSize()) jNext = 0;
    return (data[i][jNext]-data[i][jPrev])/(2.0*getDeltaTheta()*getR(i));
}


double& Mesh::getValue(const int& i, const int& j) const
{
  return data[i][j];
}

double* Mesh::getRingValues(const int& i) const
{
  return data[i];
}

void Mesh::reset() {
    for(int i=0; i<radialSize;++i)
    {
        for(int j=0; j<polarSize;++j){
            data[i][j] = 0.0;
        }
    }
}

// normalise to the area of each cell
void Mesh::normalise(){
	for(int i=0;i<getRadialSize();++i){
		for(int j=0; j<getPolarSize(); ++j)
		{
		   data[i][j]/=getDeltaR()*getR(i)*getDeltaTheta();
		}
	}
}

void Mesh::writeToFile(std::string filename) const{
	std::ofstream* out;
	out=new std::ofstream;
	out->open(filename.c_str());
	for(int i=0;i<radialSize;++i){
		for(int j=0; j<polarSize; ++j)
		{
			*out<<std::setprecision(20)<<data[i][j];
            if(j<polarSize-1) *out << ","<< std::flush;
		}
		*out<<std::endl;
	}
    out->close();
	delete out;
}




#ifndef POLARBEAMREPRESENTATION
#define POLARBEAMREPRESENTATION

class PolarBeamRepresentation {
public:
	PolarBeamRepresentation(int npart);
    virtual ~PolarBeamRepresentation();
    
    double getCharge(int i);
    double getRadius(int i);
    double getAngle(int i);
    int getSize();

    void getField(int i,double* x,double* y,double* Ex,double* Ey,double radialField, double polarField);

	void update(double* x,double* y,double* charge,int n);
	void update(double* x,double* y,int n);
protected:
    int npart;
    double* radius;
    double* angle;
    double* charge;
};

#endif

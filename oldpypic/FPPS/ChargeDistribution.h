#ifndef CHARGEDISTRIBUTION
#define CHARGEDISTRIBUTION

#include "Mesh.h"
#include "PolarBeamRepresentation.h"

class ChargeDistribution {
public:
	ChargeDistribution(Mesh* mesh);
    virtual ~ChargeDistribution();
    Mesh* getMesh();

	void fill(PolarBeamRepresentation* polarBeamRepresentation);
protected:
	virtual void addParticle(const double& r,const double& theta, const double& charge);
    Mesh* mesh;
};

#endif

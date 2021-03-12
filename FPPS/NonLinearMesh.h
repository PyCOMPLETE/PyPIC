#ifndef NLMESH
#define NLMESH

#include "Mesh.h"
#include "ChangeCoord.h"

class NonLinearMesh : public Mesh {
	public:
		NonLinearMesh(const int& n_theta, const int& m_r, const double& r_max,ChangeCoord* changeCoord);
		virtual ~NonLinearMesh();

	    virtual double computeR(const double& r) const;

        virtual double getLaplaceUpper(const int& i) const;
        virtual double getLaplaceDiag(const int& n,const int& i) const;
        virtual double getLaplaceLower(const int& i) const;

        virtual double getGradR(const int& i, const int& j) const;
        virtual double getGradTheta(const int& i, const int& j) const;
        virtual void normalise();

    protected:
        ChangeCoord* changeCoord;
};

#endif

#ifndef MESH
#define MESH

#include <string>

class Mesh{
	public:
		Mesh(const int& n_theta, const int& m_r, const double& r_max);
		virtual ~Mesh();

		int getPolarSize() const;
		int getRadialSize() const;
        int getSize() const;
		double getMaxRadius() const;
		double getR(const int& i) const;
		double getTheta(const int& i) const;

        double getDeltaR() const;
        double getDeltaTheta() const;

        virtual double getLaplaceUpper(const int& i) const;
        virtual double getLaplaceDiag(const int& n,const int& i) const;
        virtual double getLaplaceLower(const int& i) const;

        virtual double getGradR(const int& i, const int& j) const;
        virtual double getGradTheta(const int& i, const int& j) const;

		int which_Theta(const double& theta) const;
		int which_R(const double& r) const;

	    virtual double computeR(const double& r) const;
	    double computeTheta(const double& x, const double& y) const;

        double& getValue(const int& i,const int& j) const;
        double* getRingValues(const int& r) const;
        void setValue(const int& i,const int& j, double value);

        void reset();
        virtual void normalise();

	    void writeToFile(std::string filename) const;

	protected:

		int polarSize;
		int radialSize;
		double maxRadius;

	    double** data;
	    double* internalData;

};

#endif

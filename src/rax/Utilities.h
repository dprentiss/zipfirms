/////////////////////////
//
//	Utilities.h
//
/////////////////////////

#ifndef UTILITIES
#define UTILITIES

#include "time.h"

const bool UseRandomSeed = true;
const int NonRandomSeed = 20201;

///////////////////////////
//	Function prototypes...
///////////////////////////

float fmin(float, float);
float fmax(float, float);

int imin(int, int);
int imax(int, int);

bool Odd(int);

////////////////////////
//	Class definitions...
////////////////////////

class timer {
	clock_t startTime;
public:
	timer(int s);	
	void SetStartTime();														//	starts the timer
	long GetStartTime() {return startTime;};				//	returns the starting time
	float ElapsedTime();														//	returns the time difference from when this was called and the startTime
};

class RandomNumberGenerator {
	  int last;
	public:
	  RandomNumberGenerator();
		//
	  //	Return an INTEGER-valued random number in the interval (0, INT_MAX - 1)
		//
	  int Integer();
		//
	  //	Return an INTEGER-valued random number in the interval [min, max]
		//
	  int IntegerInRange(int, int);
		//
	  //	Return a REAL-valued random number in the interval [0, 1]
		//
	  float UnitReal();
		//
	  //	Return a REAL-valued random number in the interval [min, max]
		//
	  float RealInRange(float, float);
		//
		//	Return a REAL-valued, beta distributed random number on the unit interval, with parameters a and b
		//
    float UnitNormal();
    float Normal(double mean, double sd);
		float Beta1(int a, int b);
		float Beta2(int a, int b);
		float Beta(float a, float b);
		float Beta3(float a, float b);
};

const int DataVectorSize = 1000;

class Data {																					//	Size: 40 bytes
	long N;																							//		8 bytes
	double min;																					//		8 bytes
	double max;																					//		8 bytes
	double sum;																					//		8 bytes
	double sum2;																				//		8 bytes
public:
	Data();
	Data(double firstDatuum);
	void AddDatuum (double Datuum);
	long GetN() {return N;};
	double GetMin() {return min;};
	double GetMax() {return max;};
	double GetRange() {return max - min;};
	double GetAverage();
	double GetVariance();
	double GetStdDev();
	void Clear() {N = 0; min = 1000000.0; max = 0.0; sum = 0.0; sum2 = 0.0;};
	double GetSum() {return sum;};
	double GetSum2() {return sum2;};
	void CombineData (Data data1, Data data2);
};																											//	Total:  40 bytes	}

class DataVector
{																											//	Size:																								
	Data data[DataVectorSize];													//	DataVectorSize * 40 bytes; should make this into a C++ 'vector' type
public:									
	DataVector();																							
	double L2StdDev();
	double LinfStdDev();																							
};																										//	Total:	DataVectorSize * 40 bytes									
																											//	Example:	DataVectorSize = 100, size = 4K bytes
#endif

// End Utilities.h
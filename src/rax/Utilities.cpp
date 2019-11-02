//////////////////////////
//
//	Utilities.cpp
//
//////////////////////////


/////////////////
//	Includes...
/////////////////

#include <cmath>
#include <climits>

#include "Utilities.h"


/////////////////
//  Constants
/////////////////
const double Pi = 3.14159265358979323846264338;


/////////////////
//	Routines...
/////////////////

float fmin(float arg1, float arg2)
{
	if (arg1 < arg2)
		return arg1;
	else
		return arg2;
}

float fmax (float arg1, float arg2)
{
	if (arg1 > arg2)
		return arg1;
	else
		return arg2;
}

int imin(int arg1, int arg2)
{
  if (arg1 < arg2)
    return arg1;
  else
    return arg2;
};

int imax(int arg1, int arg2)
{
  if (arg1 > arg2)
    return arg1;
  else
    return arg2;
};

bool Odd(int theNumber)
{
  if ((theNumber / 2) * 2 == theNumber)
    return false;
  else
    return true;
};


//	Methods...

timer::timer(int s):
startTime(s)
{}

void timer::SetStartTime() {
	startTime = clock();
}

float timer::ElapsedTime() {
	return float(clock() - startTime)/CLOCKS_PER_SEC;
}

RandomNumberGenerator::RandomNumberGenerator():
last(0)
{
  if (UseRandomSeed)
    last = (int)time(NULL);
  else
    last = NonRandomSeed;
};  //	RandomNumberGenerator.RandomNumberGenerator()

int RandomNumberGenerator::Integer() {
	//
  //	This method generates INTEGER-valued random numbers in the interval [0, INT_MAX - 1]
  //	Source: Bratley, Fox and Schrage, 1987
	//
  int k = last / 127773;

  last = (last - k * 127773) * 16807 - k * 2836;
  if (last < 0)
    last += INT_MAX;
  return last;
}  //	RandomNumberGenerator.Integer()

int RandomNumberGenerator::IntegerInRange(int min, int max) {
	//
  //	This method generates INTEGER-valued random numbers in the interval [min, max]
	//
  return (min + Integer() % (max - min + 1));
};  //	RandomNumberGenerator.IntegerInRange()

float RandomNumberGenerator::UnitReal() {
	//
  //	This method generates REAL-valued random numbers in the interval [0, 1]
	//
  return ((float)Integer() / INT_MAX);
}  //	RandomNumberGenerator.UnitReal()

float RandomNumberGenerator::RealInRange(float min, float max) {
	//
  //	This method generates REAL-valued random numbers in the interval [min, max]
	//
  return (min + (max - min) * UnitReal());
}  //	RandomNumberGenerator.RealInRange()

float RandomNumberGenerator::UnitNormal() {
  //
  //  Well-known Box-Mueller approach
  //
  return sqrt(-2.0 * log(UnitReal())) * cos(2.0 * Pi * UnitReal());
}

float RandomNumberGenerator::Normal(double mean, double sd) {
  return mean + UnitNormal() * sd;
}


float RandomNumberGenerator::Beta1(int a, int b) {
	//
	//	This method only works for integer a and b; need to generalize this
	//
	float term1 = 1.0F;
	float term2 = 1.0F;
	int i;
	
	for (i = 1; i <= a; i++)
		term1 *= UnitReal();
	term1 = logf(term1);
	
	for (i = 1; i <= b; i++)
		term2 *= UnitReal();
	term2 = logf(term2);
	
	return term1/(term1 + term2);
}

float RandomNumberGenerator::Beta2(int a, int b) {
	//
	//	This method only works for integer a and b; need to generalize this
	//
	float term1 = 0.0F;
	float term2 = 0.0F;
	int i;
	
	for (i = 1; i <= a; i++)
		term1 += logf(UnitReal());
	
	for (i = 1; i <= b; i++)
		term2 += logf(UnitReal());
	
	return term1/(term1 + term2);
}

float RandomNumberGenerator::Beta(float a, float b)
{
	// Following Bratley, Fox and Schrage...
	//
	if ((a < 1.0F) && (b < 100.0F)) {
		float x, y;
		do {
			x = powf(UnitReal(), 1.0F/a);
			y = powf(UnitReal(), 1.0F/b);
		} while (x + y > 1.0);
		return x/(x + y);
	}
	else {
		const float log4 = logf(4.0F);
		float x, y, v, w;
		float alpha = a + b;
		float beta = fmin(a, b);
		if (beta <= 1.0)
			beta = 1.0F / beta;
		else
			beta = sqrtf((alpha - 2.0F)/(2.0F * a * b - alpha));
		float gamma = a + 1.0F / beta;
		do {
			x = UnitReal();
			y = UnitReal();
			v = beta * logf(x / (1.0F - x));
			w = a * expf(v);
		} while (alpha * logf(alpha/(b + w)) + gamma * v - log4 < logf(x * x * y));
		return w/(b + w);
	};
}

Data::Data():
	N(0), min(1000000.0), max(0.0), sum(0.0), sum2(0.0)
{}

Data::Data(double firstDatuum):
	N(1), min(firstDatuum), max(firstDatuum), sum(firstDatuum), sum2(firstDatuum * firstDatuum)
{}

void	Data::AddDatuum (double Datuum) {
	N = N + 1;
	if (Datuum < min)
		min = Datuum;
	if (Datuum > max)
		max = Datuum;
	sum += Datuum;
	sum2 += Datuum * Datuum;
}

double	Data::GetAverage() {
	if (N > 0)
		return sum / N;
	else
		return 0.0;
}

double	Data::GetVariance() {
	if (N > 1)
	{
		double avg = GetAverage();
		double arg = sum2 - N * avg * avg;
		return arg / (N - 1);
	}
	else
		return 0.0;
}

double	Data::GetStdDev() {
	double var = GetVariance();
	
	if (var > 0.0)
		return sqrt(GetVariance());
	else
		return 0.0;
}	//	Data::GetStdDev()

void Data::CombineData(Data data1, Data data2)
{
	N = data1.GetN() + data2.GetN();
	min = fmin(data1.GetMin(), data2.GetMin());
	max = fmax(data1.GetMax(), data2.GetMax());
	sum = data1.GetSum() + data2.GetSum();
	sum2 = data1.GetSum2() + data2.GetSum2();
}	//	Data::CombineData()

DataVector::DataVector() {	//	Initializer: doesn't do much at present since all the initizlization is performed in the components
	for (int i = 0; i < DataVectorSize; ++i)
		;
}	//	CommodityData::Clear()

double	DataVector::L2StdDev() {
	double sum = 0.0;
	// Instead of computing the standard deviation for each commodity
	//	(and thus calling sqrt() N times), sum the individual variances
	//	and from this get the standard deviation
	for (int i = 0; i < DataVectorSize; ++i)
		sum += data[i].GetVariance();
	return sqrt(sum);
}	//	DataVector::L2StdDev

double	DataVector::LinfStdDev() {
	double max = 0.0;
	double var;	
	// Instead of computing the standard deviation for each commodity
	//	(and thus calling sqrt() N times), find the largest variance
	//	and from it get the standard deviation
	for (int i = 0; i < DataVectorSize; ++i)
	{
		var = data[i].GetVariance();
		if (var > max)
			max = var;
	};
	return sqrt(max);
}

//	End Utilities.cpp
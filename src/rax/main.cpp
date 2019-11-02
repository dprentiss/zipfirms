//
//  Worker Exchange
//
//  Created on 3/21/16, derived from the 'Firm Kinetics' code
//
//  Basic idea here is to pick workers at random and
//    (1) move them to bigger firms with probability p or
//    (2) move them to smaller firms with probability q
//
//  In order to accomplish this we need to have populations of both firms and workers. The only thing the
//  workers do is point at the firm where they work. The firms  keep track of their size and don't need
//  to know the actual workers they employ, since we are not computing anything about production. At the
//  end we check that the right number of workers are pointing at each firm.
//

#include <iostream>
#include <cmath>

#include "Utilities.h"

//
//  First, fix some constants
//
const int numWorkers = 2000000;             //  2 x 10^6
const int numWorkersM1 = numWorkers - 1;

const int numFirms = 100000;                //  10^5, thus average firm size = 20
const int numFirmsM1 = numFirms - 1;

const long maxTime = 100000;             //  10^5: average number of activations/firm

const double p = 1.00;                   //  Probability a smaller firm loses a worker to a larger firm
const double q = 0.50;                   //  Probability a larger firm loses a worker to a smaller firm

const char *SizeDataFileName = "Sizes";
const char *GrowthDataFileName = "Growth";

const int oldGrowthLag = 10;
const bool storeOldGrowth = false;

const bool checkDataStructures = false;

class Firm {
  int old_size;
  int last_size;
  int size;
public:
  Firm();
  void SetOldSize(int theSize) {old_size = theSize;};
  int GetOldSize() {return old_size;};
  void SetLastSize(int theSize) {last_size = theSize;};
  int GetLastSize() {return last_size;};
  void SetSize(int theSize) {size = theSize;};
  int GetSize() {return size;};
  void Decrement() {size--;};
  void Increment() {size++;};
};

typedef Firm *FirmPtr;

RandomNumberGenerator RNG;

class FirmPop {
  FirmPtr firms[numFirms];
public:
  FirmPop();
  FirmPtr GetFirmPtr(int index) {return firms[index];};
  FirmPtr GetRandomFirm() {return firms[RNG.IntegerInRange(0,numFirmsM1)];};
  void SetOldSizes();
  void SetLastSizes();
  int GetMinFirmSize();
  int GetMaxFirmSize();
  int CountFirmsSize(int s);
  int CountWorkers();
} theFirms;

//
//  Methods
//
Firm::Firm() {
  old_size = 0;
  last_size = 0;
  size = 0;
}

FirmPop::FirmPop() {
  for (int i = 0; i < numFirms; i++)
    firms[i] = new Firm;
}

void FirmPop::SetOldSizes() {
  for (int i = 0; i < numFirms; i++)
    firms[i]->SetOldSize(firms[i]->GetSize());
}

void FirmPop::SetLastSizes() {
  for (int i = 0; i < numFirms; i++)
    firms[i]->SetLastSize(firms[i]->GetSize());
}

int FirmPop::GetMinFirmSize() {
  int min = 1000000;
  for (int i = 0; i < numFirms; i++)
    if (firms[i]->GetSize() < min)
      min = firms[i]->GetSize();
  return min;
}

int FirmPop::GetMaxFirmSize() {
  int max = 0;
  for (int i = 0; i < numFirms; i++)
    if (firms[i]->GetSize() > max)
      max = firms[i]->GetSize();
  return max;
}

int FirmPop::CountFirmsSize(int s) {
  int count = 0;
  for (int i = 0; i < numFirms; i++)
    if (firms[i]->GetSize() == s)
      count++;
  return count;
}

int FirmPop::CountWorkers() {
  int sum = 0;
  for (int i = 0; i < numFirms; i++)
    sum += firms[i]->GetSize();
  return sum;
}

using namespace std;

int main(int argc, const char * argv[])
{
  //
  //  This is the main array of objects
  //
  FirmPtr Workers[numWorkers];
  
  //
  //  We'll need these objects from 'Utilities'
  //
  timer TheTime(0);
  
  //
  //  Some local variables
  //
  int i;
  long time;
  int randomWorkerIndex = 0;
  FirmPtr firm1, firm2;
  bool integrityFlag = true;
  
  //
  //  A bit of IO to the CONSOLE
  //
  cout << "WORKER EXCHANGE\nRob Axtell, Krasnow Institute for Advanced Study\n" << endl;

  //
  //  Initialization code...
  //
  for (i = 0; i < numWorkers; i++) {
    firm1 = theFirms.GetRandomFirm();
    Workers[i] = firm1;
    firm1->Increment();
  }
  cout << "Total size at start: " << theFirms.CountWorkers() << "; min size: " << theFirms.GetMinFirmSize() << "; max size: " << theFirms.GetMaxFirmSize() << endl;
  
  //
  //  OK, time to get this show on the road
  //
  TheTime.SetStartTime();
  
  //
  //  One unit of time in the model is each agent activated once on average
  //
  for (time = 0; time < maxTime; time++) {
    for (i = 0; i < numWorkers; i++) {
      //
      //  Pick two firms at random and transfer a worker from one to the other:
      //    1. The firms can be picked either with uniform probability over all firms or over all workers
      //    2. Don't worry if it's the same firm, it doesn't matter;
      //
      //  Uncomment the following line to select a random firm to give up a worker:
      //
      firm1 = theFirms.GetRandomFirm();
      //
      //  Uncomment the following 2 lines to select a random worker and then use its firm to give up a worker:
      //
      //randomWorkerIndex = RNG.IntegerInRange(0, workersM1);
      //firm1 = Workers[randomWorkerIndex];
      //
      //  Uncomment the following line to select a random firm to accept the new worker:
      //
      firm2 = theFirms.GetRandomFirm();
      //
      //  Uncomment the following line to to select a random worker and then use its firm to accept the new worker:
      //
      //firm2 = Workers[RNG.IntegerInRange(0, workersM1)];
      //
      //  If the worker's firm can afford to give up the worker then we are off to the races...
      //
      if (firm1->GetSize() > 1) {
        //
        //  Either the first firm is smaller, in which case the worker moves with probability p
        //
        if (firm1->GetSize() <= firm2->GetSize()) {
          if (RNG.UnitReal() < p) {
            firm1->Decrement();
            firm2->Increment();
            //
            //  Finally, 'move' the worker to the bigger firm
            //
            Workers[randomWorkerIndex] = firm2;
          }
        }
        else
          //  ...or the first firm is bigger, in which case it moves with probability q
          //
          if (RNG.UnitReal() < q) {
            firm1->Decrement();
            firm2->Increment();
            //
            //  'Move' the worker to the smaller firm
            //
            Workers[randomWorkerIndex] = firm2;
          }
      } //  Close up test of whether giver has anything to give
      
    //
    //  Keep track of size changes in last_sizes and/or in old_sizes
    //
    }
    if (time != maxTime - 1)
      theFirms.SetLastSizes();
    if (storeOldGrowth && (time % oldGrowthLag) == 0)
      theFirms.SetOldSizes();
  } //  Close up main loop

  //
  //  Prints some stats to the CONSOLE
  //
  cout << "Total size at end:   " << theFirms.CountWorkers() << "; min size: " << theFirms.GetMinFirmSize() << "; max size:   " << theFirms.GetMaxFirmSize() << endl;
  cout << "Size 1 firms: " << theFirms.CountFirmsSize(1) << endl;
  cout << "Elapsed time: " << TheTime.ElapsedTime() << endl;
  
  //
  //  Let's check that the number of workers pointing at each firm is the same as the size of the firm
  //  Note: this routine, as presently written, wrecks data file storage by zeroing out firm sizes...
  //
  if (checkDataStructures) {
    for (i = 0; i < numWorkers; i++)
      Workers[i]->Decrement();
    for (i = 0; i < numFirms; i++)
      if (theFirms.GetFirmPtr(i)->GetSize() != 0)
        integrityFlag = false;
    if (!integrityFlag)
      cout << "***** Corrupted data structures!! *****" << endl;
  }
  std::cout << std::endl;
  
  //
  //  Store raw data on firm sizes
  //
  FILE *DataFile1 = fopen(SizeDataFileName, "w");
	putc('{', DataFile1);
  for (i = 0; i < numFirms; i++) {
    fprintf(DataFile1, "%8d", theFirms.GetFirmPtr(i)->GetSize());
    if (i != numFirmsM1)
      fprintf(DataFile1, ", ");
    if ((i % 10) == 9)
      fprintf(DataFile1, "\n");
  }
	putc('}', DataFile1);
	fclose(DataFile1);

  //
  //  Lastly, store raw data on firm growth rate
  //
  FILE *DataFile2 = fopen(GrowthDataFileName, "w");
	putc('{', DataFile2);
  for (i = 0; i < numFirms; i++) {
    if (!storeOldGrowth) {
      if (theFirms.GetFirmPtr(i)->GetLastSize() > 0) {
        fprintf(DataFile2, "%8.3f", (double)theFirms.GetFirmPtr(i)->GetSize() / (double)theFirms.GetFirmPtr(i)->GetLastSize());
        if (i != numFirmsM1)
          fprintf(DataFile2, ", ");
      }
    }
    else
      if (theFirms.GetFirmPtr(i)->GetOldSize() > 0) {
        fprintf(DataFile2, "%8.3f", (double)theFirms.GetFirmPtr(i)->GetSize() / (double)theFirms.GetFirmPtr(i)->GetOldSize());
        if (i != numFirmsM1)
          fprintf(DataFile2, ", ");
      }
    if ((i % 10) == 9)
      fprintf(DataFile2, "\n");
  }
	putc('}', DataFile2);
	fclose(DataFile2);
  
  return 0;
}
#include <cstdint>
#include <cmath>
extern "C" void weighted_ap(int64_t n, int64_t draws, int64_t units,
 const int32_t* group, const uint8_t* positive, const uint8_t* end,
 const int32_t* weights, double* result) {
 #pragma omp parallel for schedule(static)
 for(int64_t d=0;d<draws;d++) {
  const int32_t* w=weights+d*units;
  double total=0, tp=0, tied_tp=0, numerator=0;
  for(int64_t i=0;i<n;i++) {
   const double weight=w[group[i]];
   total+=weight;
   if(positive[i]) {tp+=weight; tied_tp+=weight;}
   if(end[i]) {
    if(total>0) numerator+=tied_tp*tp/total;
    tied_tp=0;
   }
  }
  result[d]=tp>0 ? numerator/tp : NAN;
 }
}

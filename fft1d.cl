
#include "fft_8.cl" 

#include "../host/inc/fft_config.h" 

kernel void fft1d(global float2 * restrict src, global float2 * restrict dest,
                  int count, int inverse) {

  const int N = (1 << LOGN);

  float2 fft_delay_elements[N + 8 * (LOGN - 2)];

  for (unsigned i = 0; i < count * (N / 8) + N / 8 - 1; i++) {

    int base = (i / (N / 8)) * N;
    int offset = i % (N / 8);

    float2x8 data;
    
    if (i < count * (N / 8)) {
      data.i0 = src[base + offset];
      data.i1 = src[base + 4 * N / 8 + offset];
      data.i2 = src[base + 2 * N / 8 + offset];
      data.i3 = src[base + 6 * N / 8 + offset];
      data.i4 = src[base + N / 8 + offset];
      data.i5 = src[base + 5 * N / 8 + offset];
      data.i6 = src[base + 3 * N / 8 + offset];
      data.i7 = src[base + 7 * N / 8 + offset];
    } else {
      data.i0 = data.i1 = data.i2 = data.i3 = 
                data.i4 = data.i5 = data.i6 = data.i7 = 0;
    }

    data = fft_step(data, i % (N / 8), fft_delay_elements, inverse, LOGN); 

    if (i >= N / 8 - 1) {
      int base = 8 * (i - (N / 8 - 1));
 
      dest[base] = data.i0;
      dest[base + 1] = data.i1;
      dest[base + 2] = data.i2;
      dest[base + 3] = data.i3;
      dest[base + 4] = data.i4;
      dest[base + 5] = data.i5;
      dest[base + 6] = data.i6;
      dest[base + 7] = data.i7;
    }
  }
}

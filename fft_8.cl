#include "twid_radix4_8.cl" 

typedef struct {
   float2 i0;
   float2 i1;
   float2 i2;
   float2 i3;
   float2 i4;
   float2 i5;
   float2 i6;
   float2 i7;
} float2x8;

float2x8 butterfly(float2x8 data) {
   float2x8 res;
   res.i0 = data.i0 + data.i1;
   res.i1 = data.i0 - data.i1;
   res.i2 = data.i2 + data.i3;
   res.i3 = data.i2 - data.i3;
   res.i4 = data.i4 + data.i5;
   res.i5 = data.i4 - data.i5;
   res.i6 = data.i6 + data.i7;
   res.i7 = data.i6 - data.i7;
   return res;
}

float2x8 swap_complex(float2x8 data) {
   float2x8 res;
   res.i0.x = data.i0.y;
   res.i0.y = data.i0.x;
   res.i1.x = data.i1.y;
   res.i1.y = data.i1.x;
   res.i2.x = data.i2.y;
   res.i2.y = data.i2.x;
   res.i3.x = data.i3.y;
   res.i3.y = data.i3.x;
   res.i4.x = data.i4.y;
   res.i4.y = data.i4.x;
   res.i5.x = data.i5.y;
   res.i5.y = data.i5.x;
   res.i6.x = data.i6.y;
   res.i6.y = data.i6.x;
   res.i7.x = data.i7.y;
   res.i7.y = data.i7.x;
   return res;
}

float2x8 trivial_rotate(float2x8 data) {
   float2 tmp = data.i3;
   data.i3.x = tmp.y;
   data.i3.y = -tmp.x;
   tmp = data.i7;
   data.i7.x = tmp.y;
   data.i7.y = -tmp.x;
   return data;
}

float2x8 trivial_swap(float2x8 data) {
   float2 tmp = data.i1;
   data.i1 = data.i2;
   data.i2 = tmp;
   tmp = data.i5;
   data.i5 = data.i6;
   data.i6 = tmp;
   return data;
}

float2x8 swap(float2x8 data) {
   float2 tmp = data.i1;
   data.i1 = data.i4;
   float2 tmp2 = data.i2;
   data.i2 = tmp;
   tmp = data.i3;
   data.i3 = data.i5;
   data.i4 = tmp2;
   data.i5 = data.i6;
   data.i6 = tmp;
   return data;
}


float2 delay(float2 data, const int depth, float2 *shift_reg) {
   shift_reg[depth] = data;
   return shift_reg[0];
}

float2x8 reorder_data(float2x8 data, const int depth, float2 * shift_reg, bool toggle) {
   data.i1 = delay(data.i1, depth, shift_reg);
   data.i3 = delay(data.i3, depth, shift_reg + depth + 1);
   data.i5 = delay(data.i5, depth, shift_reg + 2 * (depth + 1));
   data.i7 = delay(data.i7, depth, shift_reg + 3 * (depth + 1));
 
   if (toggle) {
      float2 tmp = data.i0;
      data.i0 = data.i1;
      data.i1 = tmp;
      tmp = data.i2;
      data.i2 = data.i3;
      data.i3 = tmp;
      tmp = data.i4;
      data.i4 = data.i5;
      data.i5 = tmp;
      tmp = data.i6;
      data.i6 = data.i7;
      data.i7 = tmp;
   }

   data.i0 = delay(data.i0, depth, shift_reg + 4 * (depth + 1));
   data.i2 = delay(data.i2, depth, shift_reg + 5 * (depth + 1));
   data.i4 = delay(data.i4, depth, shift_reg + 6 * (depth + 1));
   data.i6 = delay(data.i6, depth, shift_reg + 7 * (depth + 1));

   return data;
}

float2 comp_mult(float2 a, float2 b) {
   float2 res;
   res.x = a.x * b.x - a.y * b.y;
   res.y = a.x * b.y + a.y * b.x;
   return res;
}

float2 twiddle(int index, int stage, int size, int stream) {
   float2 twid;
   constant float * twiddles_cos[TWID_STAGES][6] = {
                        {tc00, tc01, tc02, tc03, tc04, tc05}, 
                        {tc10, tc11, tc12, tc13, tc14, tc15}, 
                        {tc20, tc21, tc22, tc23, tc24, tc25}, 
                        {tc30, tc31, tc32, tc33, tc34, tc35}, 
                        {tc40, tc41, tc42, tc43, tc44, tc45}
   };
   constant float * twiddles_sin[TWID_STAGES][6] = {
                        {ts00, ts01, ts02, ts03, ts04, ts05}, 
                        {ts10, ts11, ts12, ts13, ts14, ts15}, 
                        {ts20, ts21, ts22, ts23, ts24, ts25}, 
                        {ts30, ts31, ts32, ts33, ts34, ts35}, 
                        {ts40, ts41, ts42, ts43, ts44, ts45}
   };

   int twid_stage = stage >> 1;
   if (size <= (1 << (TWID_STAGES * 2 + 2))) {
      twid.x = twiddles_cos[twid_stage][stream]
                                  [index * ((1 << (TWID_STAGES * 2 + 2)) / size)];
      twid.y = twiddles_sin[twid_stage][stream]
                                  [index * ((1 << (TWID_STAGES * 2 + 2)) / size)];
   } else {
      const float TWOPI = 2.0f * M_PI_F;
      int multiplier;
      
      int phase = 0;
      if (stream >= 3) {
         stream -= 3; 
         phase = 1;
      }
      switch (stream) {
         case 0: multiplier = 2; break;
         case 1: multiplier = 1; break;
         case 2: multiplier = 3; break;
         default: multiplier = 0;
      }
      int pos = (1 << (stage - 1)) * multiplier * ((index + (size / 8) * phase) 
                                          & (size / 4 / (1 << (stage - 1)) - 1));
      float theta = -1.0f * TWOPI / size * (pos & (size - 1));
      twid.x = cos(theta);
      twid.y = sin(theta);
   }
   return twid;
}

float2x8 complex_rotate(float2x8 data, int index, int stage, int size) {
   data.i1 = comp_mult(data.i1, twiddle(index, stage, size, 0));
   data.i2 = comp_mult(data.i2, twiddle(index, stage, size, 1));
   data.i3 = comp_mult(data.i3, twiddle(index, stage, size, 2));
   data.i5 = comp_mult(data.i5, twiddle(index, stage, size, 3));
   data.i6 = comp_mult(data.i6, twiddle(index, stage, size, 4));
   data.i7 = comp_mult(data.i7, twiddle(index, stage, size, 5));
   return data;
}

float2x8 fft_step(float2x8 data, int step, float2 *fft_delay_elements, 
                  bool inverse, const int logN) {
    const int size = 1 << logN;

    if (inverse) {
       data = swap_complex(data);
    }

    data = butterfly(data);
    data = trivial_rotate(data);
    data = trivial_swap(data);

    #pragma unroll
    for (int stage = 1; stage < logN - 2; stage++) {
        bool complex_stage = stage & 1; // stages 3, 5, ...

        int data_index = (step + ( 1 << (logN - 1 - stage))) & (size / 8 - 1);

        data = butterfly(data);

        if (complex_stage) {
            data = complex_rotate(data, data_index, stage, size);
        }

        data = swap(data);

  
        int delay = 1 << (logN - 2 - stage);

        
        bool toggle = data_index & delay;

        
        float2 *head_buffer = fft_delay_elements + 
                              size - (1 << (logN - stage + 2)) + 8 * (stage - 2);

        data = reorder_data(data, delay, head_buffer, toggle);

        if (!complex_stage) {
            data = trivial_rotate(data);
        }
    }

    data = butterfly(data);
    data = complex_rotate(data, step & (size / 8 - 1), 1, size);
    data = swap(data);


    data = butterfly(data);

    #pragma unroll
    for (int ii = 0; ii < size + 8 * (logN - 2) - 1; ii++) {
        fft_delay_elements[ii] = fft_delay_elements[ii + 1];
    }

    if (inverse) {
       data = swap_complex(data);
    }

    return data;
}


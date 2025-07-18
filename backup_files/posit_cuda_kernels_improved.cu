#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <stdio.h>

#include <fstream>

#define FP16_LIMB_SIZE 16
#define FP16_TYPE uint16_t



#define SIGN_MASK 0x8000
#define FLOAT_SIGN_MASK 0x80000000
#define FLOAT_SIGN_RESET_MASK 0x7FFFFFFF
#define SECOND_BIT_MASK 0x4000
#define POSIT_INF 0x0000
#define POSIT_LIMB_ALL_BITS_SET 0xffff
#define SINGLE_PRECISION_BIAS 127
#define FLOAT_SIZE 32
#define FLOAT_EXPONENT_MASK 0x7f800000
#define FLOAT_FRACTION_MASK 0x007fffff
#define FLOAT_SIGN_SHIFT 31
#define FLOAT_EXPONENT_SHIFT 23
#define FLOAT_DENORMAL_EXPONENT -126
#define FLOAT_HIDDEN_BIT_SET_MASK 0x00800000
#define FLOAT_SIGN_PLUS_EXP_LENGTH_MINUS_ONE 8
#define TEMP_TYPE uint64_t
#define UNSIGNED_LONG_LONG_SIZE 64
#define EDP_ACC_SIZE 63
#define POSIT_EXP_SHIFT 41 //64-23
#define FLOAT_EXP_SIGN_SHIFT 30
#define FLOAT_INF 0x7F800000
#define FLOAT_SIGN_PLUS_EXP_LENGTH 9
#define POSIT_LENGTH_PLUS_ONE 17

#define GET_MAX(a, b)                                                          \
  ({                                                                           \
    __typeof__(a) _a = (a);                                                    \
    __typeof__(b) _b = (b);                                                    \
    _a > _b ? _a : _b;                                                         \
  })

#define _G_INFP 32768

union Bits {
	float f;
	int32_t si;
	uint32_t ui;
};

typedef FP16_TYPE fp16;

/* things need to define
#if _G_NBITS == 16
#define _G_POSIT_SHIFT_AMOUNT 0
#define _G_MAXREALP 32767
#define _G_MINREALP 1
#define POSIT_EXTRA_BITS_SHIFT 49 // 64 - _G_NBITS + 1
#define POSIT_EXTRA_BITS_MASK 0x0000FFFFFFFFFFFF
#define POSIT_HALFWAY_BIT_MASK 0x0001000000000000

#if _G_ESIZE == 1
#define _G_USEED 4
#define _G_USEED_ZEROS 2
#define POSIT_EXPONENT_MASK 1
#define _G_MAXREAL 2.684354560e+8
#define _G_MINREAL 3.725290298e-9
#define _G_MAXREAL_INT 0x8D800000
#define _G_MINREAL_INT 0x31800000

*/
/*
summary:
uint32_t  size 11 : [_G_POSIT_SHIFT_AMOUNT, _G_MAXREALP, _G_MINREALP, POSIT_EXTRA_BITS_SHIFT,
              _G_USEED, _G_USEED_ZEROS, POSIT_EXPONENT_MASK, _G_MAXREAL_INT, _G_MINREAL_INT, _G_NBITS,_G_ESIZE]

uint64_t size 2 : POSIT_EXTRA_BITS_MASK, POSIT_HALFWAY_BIT_MASK
*/
#define _G_POSIT_SHIFT_AMOUNT   int32_constants[0]
#define _G_MAXREALP             int32_constants[1]
#define _G_MINREALP             int32_constants[2]
#define POSIT_EXTRA_BITS_SHIFT  int32_constants[3]
#define _G_USEED                int32_constants[4]
#define _G_USEED_ZEROS          int32_constants[5]
#define POSIT_EXPONENT_MASK     int32_constants[6]
#define _G_MAXREAL_INT          int32_constants[7]
#define _G_MINREAL_INT          int32_constants[8]
#define _G_NBITS                int32_constants[9]
#define _G_ESIZE                int32_constants[10]

#define POSIT_EXTRA_BITS_MASK   int64_constants[0]
#define POSIT_HALFWAY_BIT_MASK  int64_constants[1]
__constant__ uint32_t	int32_constants[ 11 ];
__constant__ uint64_t	int64_constants[ 2 ];

__device__ __inline__ float fp16tofp32_gpu(fp16 p) {
  union Bits v;

  // get sign
  bool sign = p & SIGN_MASK;
  p = (p ^ -sign) + sign;

  // get the regime sign
  bool regime_sign = p & SECOND_BIT_MASK;

  // get regime
  v.ui = p << POSIT_LENGTH_PLUS_ONE;
  //int regime_length = (__clz(v.ui) & -!regime_sign) + (__clz(~v.ui) & -regime_sign);
  int regime_length;
  if(regime_sign)
    regime_length = (__clz(~v.ui));
  else
    regime_length = (__clz(v.ui));
  int regime = (regime_length - regime_sign) << _G_ESIZE;
  regime = (regime ^ -regime_sign) + regime_sign;

  // assemble
  v.ui <<= (regime_length + 1);
  v.ui >>= (FLOAT_SIGN_PLUS_EXP_LENGTH - _G_ESIZE);
  v.ui += ((SINGLE_PRECISION_BIAS - regime) << FLOAT_EXPONENT_SHIFT);

  v.si ^= (FLOAT_INF ^ v.si) & -(p == _G_INFP);
  v.si ^= (0 ^ v.si) & -(p == 0);

  v.ui |= (sign << FLOAT_SIGN_SHIFT);
  return v.f;

}

__device__ __inline__ fp16 fp32tofp16_gpu(float f) {
  fp16 p = 0;
  union Bits v;
  v.f = f;
  bool sign = v.ui & FLOAT_SIGN_MASK;
  v.ui &= 0x7FFFFFFF;

#ifdef FLOAT_ROUNDING
	uint16_t roundSign = sign << 15;
	if(v.ui > _G_MAXREAL_INT)
		return _G_INFP | roundSign;
	if(v.ui < _G_MINREAL_INT)
		return 0;
#endif
  p ^= (p ^_G_MAXREALP) & -(v.si >= _G_MAXREAL_INT);
  p ^= (p ^ _G_INFP) & -(v.si >= FLOAT_INF);
  p ^= (p ^ _G_MINREALP) & -(v.si != 0 && v.si <= _G_MINREAL_INT);

  // min posit exponent in 16, 3 is 112
  // therefore all the float subnormals will be handled
  // in the previous if statement

  // get exponent sign
  bool exp_sign = !(v.ui >> FLOAT_EXP_SIGN_SHIFT);

  //get regime and exponent
  uint32_t exp = abs((v.si >> FLOAT_EXPONENT_SHIFT) - SINGLE_PRECISION_BIAS);
  TEMP_TYPE regime_and_exp = (((1 << ((exp >> _G_ESIZE) + 1)) - 1) << (_G_ESIZE + 1)) | (exp & POSIT_EXPONENT_MASK);;
  //if exponent is negative
  regime_and_exp = ((regime_and_exp ^ -exp_sign) + exp_sign) >> ((exp_sign & !((exp & POSIT_EXPONENT_MASK))) & (bool) exp);
  int regime_and_exp_length = (exp >> _G_ESIZE) + 2 + _G_ESIZE - ((exp_sign & !((exp & POSIT_EXPONENT_MASK))) & (bool) exp);

  //assemble
  regime_and_exp <<= (UNSIGNED_LONG_LONG_SIZE - regime_and_exp_length);
  regime_and_exp |= ((TEMP_TYPE) (v.ui & FLOAT_FRACTION_MASK) << (POSIT_EXP_SHIFT - regime_and_exp_length));
  fp16 temp_p = (regime_and_exp >> POSIT_EXTRA_BITS_SHIFT);

  //round
  temp_p += (bool) (regime_and_exp & POSIT_HALFWAY_BIT_MASK) && ((temp_p & 1) | (regime_and_exp & POSIT_EXTRA_BITS_MASK));
 if (_G_NBITS != 16)
    temp_p <<= _G_POSIT_SHIFT_AMOUNT;

  p ^= (temp_p ^ p) & -((v.si < _G_MAXREAL_INT) & (v.si > _G_MINREAL_INT));

  p = (p ^ -sign) + sign;

  return p;
}

#define MUL_SCALE 1.0
#define MUL_SCALE_WEIGHT 64.0
//template <typename scalar_t>
__global__ void posit_cuda_kernel(
     float* input,
    size_t input_size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < input_size) {
    float temp_float = input[index] * MUL_SCALE;
    fp16 temp = fp32tofp16_gpu(temp_float);
    temp_float = fp16tofp32_gpu(temp);
    temp_float = temp_float/MUL_SCALE;
    input[index] = temp_float;//fp16tofp32_gpu(temp);

  }
}
//template <typename scalar_t>
__global__ void posit_cuda_kernel_weight(
     float* input,
    size_t input_size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < input_size) {
    float temp_float = input[index] * MUL_SCALE_WEIGHT;
    fp16 temp = fp32tofp16_gpu(temp_float);
    temp_float = fp16tofp32_gpu(temp);
    temp_float = temp_float/MUL_SCALE_WEIGHT;
    input[index] = temp_float;//fp16tofp32_gpu(temp);

  }
}
void generate_constants(int nsize, int es, uint32_t* int32_constants, uint64_t* int64_constants) {
  //local vars have the same name as global constant vars, confusing but less likely error can happen here.
  //ugly but it's the traightforward conversion from the original #define macroes;
  //todo: make this one less messy
  _G_NBITS = nsize;
  _G_ESIZE = es;
  if (nsize == 16 ){
    _G_POSIT_SHIFT_AMOUNT = 0;
    _G_MAXREALP = 32767;
    _G_MINREALP =  1;
    POSIT_EXTRA_BITS_SHIFT = 49 ;// 64 - _G_NBITS + 1
    POSIT_EXTRA_BITS_MASK = 0x0000FFFFFFFFFFFF;
    POSIT_HALFWAY_BIT_MASK = 0x0001000000000000;

      switch(es) {
       case 1  :
            _G_USEED = 4;
            _G_USEED_ZEROS  = 2;
            POSIT_EXPONENT_MASK = 1;
            _G_MAXREAL_INT =  0x4D800000;
            _G_MINREAL_INT = 0x31800000;
          break; //optional
       case 2  :
            _G_USEED = 16;
            _G_USEED_ZEROS = 4;
            POSIT_EXPONENT_MASK = 3;
            _G_MAXREAL_INT = 0x5B800000;
            _G_MINREAL_INT = 0x23800000;
          break; //optional

       default : //Optional
            //no case;
            printf("unexpected posit config\n");
            exit(1);

    }

  } else if (nsize == 8){
     _G_POSIT_SHIFT_AMOUNT =  8;
     _G_MAXREALP = 32512;
     _G_MINREALP = 256;
     POSIT_EXTRA_BITS_SHIFT = 57;
     POSIT_EXTRA_BITS_MASK = 0x00FFFFFFFFFFFFFF;
     POSIT_HALFWAY_BIT_MASK = 0x0100000000000000;

    switch(es) {
     case 1  :
      _G_USEED = 4;
      _G_USEED_ZEROS = 2;
      POSIT_EXPONENT_MASK = 1;
      _G_MAXREAL_INT = 0x45800000;
      _G_MINREAL_INT = 0x39800000;
        break; //optional
     case 2  :
      _G_USEED = 16;
      _G_USEED_ZEROS = 4;
      POSIT_EXPONENT_MASK = 3;
      _G_MAXREAL_INT = 0x4B800000;
      _G_MINREAL_INT = 0x33800000;
        break; //optional

     default : //Optional
          //no case;
          printf("unexpected posit config\n");
          exit(1);
        }

  } else if (nsize == 6){
     _G_POSIT_SHIFT_AMOUNT = 10;
     _G_MAXREALP = ((1 << (_G_NBITS - 1)) - 1) << _G_POSIT_SHIFT_AMOUNT;
     _G_MINREALP = (1 << _G_POSIT_SHIFT_AMOUNT);
     POSIT_EXTRA_BITS_SHIFT =  (64 - _G_NBITS + 1);
     POSIT_EXTRA_BITS_MASK = 0x03FFFFFFFFFFFFFF;
     POSIT_HALFWAY_BIT_MASK = 0x0400000000000000;

    switch(es) {
     case 1  :
      _G_USEED = 4;
      _G_USEED_ZEROS = 2;
      POSIT_EXPONENT_MASK = 1;

      _G_MAXREAL_INT = 0x43800000;
      _G_MINREAL_INT = 0x3b800000;
        break; //optional
     case 2  :
      _G_USEED = 16;
      _G_USEED_ZEROS = 4;
      POSIT_EXPONENT_MASK = 3;
      _G_MAXREAL_INT = 0x47800000;
      _G_MINREAL_INT = 0x377ffff6;
        break; //optional

     default : //Optional
          //no case;
          printf("unexpected posit config\n");
          exit(1);
    }

  } else if (nsize == 10){
     _G_POSIT_SHIFT_AMOUNT = 6;
     _G_MAXREALP = 32704;
     _G_MINREALP = 64;
     POSIT_EXTRA_BITS_SHIFT = 55;
     POSIT_EXTRA_BITS_MASK = 0x003FFFFFFFFFFFFF;
     POSIT_HALFWAY_BIT_MASK = 0x0040000000000000;
    switch(es) {
     case 1  :
      _G_USEED = 4;
      _G_USEED_ZEROS = 2;
      POSIT_EXPONENT_MASK = 1;
      _G_MAXREAL_INT = 0x47800000;
      _G_MINREAL_INT = 0x37800000;
        break; //optional
     case 2  :
      _G_USEED = 16;
      _G_USEED_ZEROS = 4;
      POSIT_EXPONENT_MASK = 3;
      _G_MAXREAL_INT = 0x4F800000;
      _G_MINREAL_INT = 0x2F800000;
        break; //optional

     default : //Optional
          //no case;
          printf("unexpected posit config\n");
          exit(1);
      }
  }else if (nsize == 4){
     _G_POSIT_SHIFT_AMOUNT = 12;
     _G_MAXREALP = ((1 << (_G_NBITS - 1)) - 1) << _G_POSIT_SHIFT_AMOUNT;
     _G_MINREALP = (1 << _G_POSIT_SHIFT_AMOUNT);
     POSIT_EXTRA_BITS_SHIFT =  (64 - _G_NBITS + 1);
     POSIT_EXTRA_BITS_MASK = 0x0FFFFFFFFFFFFFF;
     POSIT_HALFWAY_BIT_MASK = 0x1000000000000000;
    switch(es) {
     case 1  :
      _G_USEED = 4;
      _G_USEED_ZEROS = 2;
      POSIT_EXPONENT_MASK = 1;
      _G_MAXREAL_INT = 0x41800000; // 16
      _G_MINREAL_INT = 0x3d800000; // 0.0625
        break; //optional
     case 2  :
      _G_USEED = 16;
      _G_USEED_ZEROS = 4;
      POSIT_EXPONENT_MASK = 3;
      _G_MAXREAL_INT = 0x43800000; // 256
      _G_MINREAL_INT = 0x3b800000; // 1/256
        break; //optional

     default : //Optional
          //no case;
          printf("unexpected posit config\n");
          exit(1);
      }
  } else {
    printf("unexpected posit config\n");
    exit(1);
  }
};


torch::Tensor posit_cuda(
    torch::Tensor input) {

      int nsize = 6;
      int es = 1;
//  const auto state_size = input.size(0);
  int64_t input_size = 1;
  for (int i = 0 ; i< input.sizes().size();i++)
    input_size = input_size * input.sizes()[i];

  uint32_t int32_constants_host[11];
  uint64_t int64_constants_host[2];
  generate_constants(nsize, es, int32_constants_host, int64_constants_host );
  const int threads = 1024;
  const dim3 blocks((input_size + threads - 1) / threads);

   cudaMemcpyToSymbol( int32_constants, &int32_constants_host[0], 11 * sizeof( uint32_t ), 0 );
   cudaMemcpyToSymbol( int64_constants, &int64_constants_host[0], 2 * sizeof( uint64_t ), 0 );


    posit_cuda_kernel<<<blocks, threads>>>(
    input.data<float>(),
        input_size );




  return input;
}

torch::Tensor posit_cuda_weight(
    torch::Tensor input) {

      int nsize = 4;
      int es = 1;
//  const auto state_size = input.size(0);
  int64_t input_size = 1;
  for (int i = 0 ; i< input.sizes().size();i++)
    input_size = input_size * input.sizes()[i];

  uint32_t int32_constants_host[11];
  uint64_t int64_constants_host[2];
  generate_constants(nsize, es, int32_constants_host, int64_constants_host );
  const int threads = 1024;
  const dim3 blocks((input_size + threads - 1) / threads);

   cudaMemcpyToSymbol( int32_constants, &int32_constants_host[0], 11 * sizeof( uint32_t ), 0 );
   cudaMemcpyToSymbol( int64_constants, &int64_constants_host[0], 2 * sizeof( uint64_t ), 0 );


    posit_cuda_kernel_weight<<<blocks, threads>>>(
    input.data<float>(),
        input_size );




  return input;
}

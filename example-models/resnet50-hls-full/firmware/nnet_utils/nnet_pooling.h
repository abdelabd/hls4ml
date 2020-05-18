#ifndef NNET_POOLING_H_
#define NNET_POOLING_H_

#include <iostream>
#include "nnet_helpers.h"

namespace nnet{

// Return the maximum value from an array
template<typename T, int N>
T max(T x[N]){
  T y = x[0];
  for(int i = 1; i < N; i++){
    y = x[i] > y ? x[i] : y;
  }
  return y;
}

template<int W, int N>
ap_int<W> avg(ap_int<W> (&x)[N]){
  // Use a wider accumulator than the input to avoid overflow
  ap_int<W + ceillog2(N)> tmp = 0;
  for(int i = 0; i < N; i++){
    tmp += x[i];
  }
  tmp /= N;
  // Now cast back to original type
  ap_int<W> y = tmp;
  return tmp;
}

template<int W, int I, int N>
ap_fixed<W, I> avg(ap_fixed<W, I> (&x)[N]){
  // Use a wider accumulator than the input to avoid overflow
  ap_fixed<W + ceillog2(N), I + ceillog2(N)> tmp = 0;
  for(int i = 0; i < N; i++){
    tmp += x[i];
  }
  tmp /= N;
  // Now cast back to original type
  ap_fixed<W, I> y = tmp;
  return y;
}

// Return the mean value of an array
template<typename T, int N>
T avg(T (&x)[N]){
  T y = 0;
  for(int i = 0; i < N; i++){
    y += x[i];
  }
  y /= N;
  return y;
}

// Enumeration for pooling operation (max, avg, l2norm pooling)
enum Pool_Op { Max, Average }; // L2Norm };
template<typename T, int N, Pool_Op op>
T pool_op(T (&x)[N]){
	switch(op){
	case Max: return max<T, N>(x);
	case Average: return avg(x);
	// case L2Norm: return l2norm<T, N>(x);
	}
}

template<typename T, Pool_Op op>
T pad_val(){
  /*---
  *- In Tensorflow, pooling ignores the value in the padded cells
  *- For Avg pooling, return 0 (the divisior is modified to the
  *- area overlapping the unpadded image.
  *- For max pooling, return the most negative value for the type.
  *- TODO this is not really generic, it assumes fixed point or integer T
  ---*/
  switch(op){
    case Max:{ 
      T x = 0;
      x[x.width - 1] = 1;
      return x;
      break;}
    case Average: return 0;
  }
}

struct pooling1d_config{
  // IO size
  static const unsigned n_in = 10;
  static const unsigned pool_size = 2;
  static const unsigned n_out = n_in / pool_size;
  static const unsigned pad_left = 0;
  static const unsigned pad_right = 0;
  // Pooling function
  static const Pool_Op pool_op = Max;
};

template<class data_T, typename CONFIG_T>
void pooling1d(data_T data[CONFIG_T::n_in], data_T res[CONFIG_T::n_out]){
  for(int ii = 0; ii < CONFIG_T::n_out; ii ++){
    data_T pool[CONFIG_T::pool_size];
    for(int jj = 0; jj < CONFIG_T::pool_size; jj++){
      pool[jj] = data[ii * CONFIG_T::pool_size + jj]; 
    }
    res[ii] = pool_op<data_T, CONFIG_T::pool_size, CONFIG_T::pool_op>(pool);
  }
}

struct pooling2d_config{
  // IO size
  static const unsigned in_height = 10;
  static const unsigned in_width = 10;
  static const unsigned n_filt = 4;
  static const unsigned stride_height = 2;
  static const unsigned stride_width = 2;
  static const unsigned pool_height = 2;
  static const unsigned pool_width = 2;
  static const unsigned out_height = (in_height - pool_height) / stride_height + 1;
  static const unsigned out_width = (in_width - pool_width) / stride_width + 1;
  // Padding
  static const unsigned pad_top = 0;
  static const unsigned pad_bottom = 0;
  static const unsigned pad_left = 0;
  static const unsigned pad_right = 0;
  // Pooling function
  static const Pool_Op pool_op = Max;
  // Reuse
  static const unsigned reuse = 1;
};

template<class data_T, typename CONFIG_T>
void pooling2d_filt_cl(data_T data[CONFIG_T::pool_height * CONFIG_T::pool_width * CONFIG_T::n_filt],
		       data_T res[CONFIG_T::n_filt]){
  for(unsigned i0 = 0; i0 < CONFIG_T::n_filt; i0++) { 
    #pragma HLS UNROLL
    data_T pool[CONFIG_T::pool_height * CONFIG_T::pool_width];
    #pragma HLS ARRAY_RESHAPE variable=pool complete dim=0
    for(unsigned i1 = 0; i1 < CONFIG_T::pool_height*CONFIG_T::pool_width; i1++) { 
      pool[i1] = data[i1*CONFIG_T::n_filt+i0];
    }
    res[i0] = pool_op<data_T, CONFIG_T::pool_height*CONFIG_T::pool_width, CONFIG_T::pool_op>(pool);
  }
}


template<class data_T, class res_T, typename CONFIG_T>
  void clonetmp(
		data_T input[CONFIG_T::n_chan],
		res_T  data [CONFIG_T::n_filt]) { //CONFIG_T::n_filt2
  #pragma HLS PIPELINE
  for(unsigned i2 = 0; i2 < CONFIG_T::n_filt; i2++) {
   #pragma HLS UNROLL
    data[i2] = input[i2];
  }
}

/*
template<class data_T, class res_T, typename CONFIG_T>
void pool_2d_large_stream(
		      hls::stream<data_T> data[CONFIG_T::n_chan],
		      hls::stream<res_T>  res [CONFIG_T::n_filt]) { 

    const static int lShiftX = CONFIG_T::filt_width-CONFIG_T::pad_left-1;
    const static int lShiftY = CONFIG_T::filt_height-CONFIG_T::pad_top-1;

    static data_T layer_in_row[CONFIG_T::in_width+CONFIG_T::pad_left+CONFIG_T::pad_right][CONFIG_T::filt_height][CONFIG_T::n_chan];
    #pragma HLS ARRAY_RESHAPE variable=layer_in_row complete dim=3

    static data_T layer_in[CONFIG_T::filt_height*CONFIG_T::filt_width*CONFIG_T::n_chan];
    //#pragma HLS ARRAY_RESHAPE variable=layer_in block factor=CONFIG_T::n_chan dim=0
    #pragma HLS ARRAY_RESHAPE variable=layer_in complete dim=0

    static unsigned pX=0;
    static unsigned pY=0;
    static bool pPass = false;
    if(pX == 0 && pY == 0) pPass = false;
    for(int i0 = 0; i0 < CONFIG_T::n_chan; i0++) { 
       #pragma HLS UNROLL
       layer_in_row[pX+CONFIG_T::pad_left][(pY+CONFIG_T::pad_top) % CONFIG_T::filt_height][i0] =  data[i0].read();
    }
    //Processs image
    if(pX == 0) nnet::reset_down_2dXNew<data_T,data_T,CONFIG_T>(pX,layer_in_row,layer_in); //check stride
    if((pX+1) % CONFIG_T::stride_width == 0 && (pY+1) % CONFIG_T::stride_height == 0 && pPass) { 
      for(unsigned i0 = 0; i0 < CONFIG_T::n_filt; i0++) { 
        #pragma HLS UNROLL
	data_T pool[CONFIG_T::pool_height * CONFIG_T::pool_width];
        #pragma HLS ARRAY_RESHAPE variable=pool complete dim=0
        for(unsigned i1 = 0; i1 < CONFIG_T::pool_height*CONFIG_T::pool_width; i1++) { 
         #pragma HLS UNROLL
	  pool[i1] = layer_in[i1+i0*CONFIG_T::n_filt];
	}
	res[i0].write(pool_op<data_T, CONFIG_T::pool_height*CONFIG_T::pool_width, CONFIG_T::pool_op>(pool));
      }
      nnet::shift_right_stride_2dNew<data_T,data_T,CONFIG_T>(pX,pY,layer_in_row,layer_in);//check stride
    }    
    pX = pX+1;
    if(pX == CONFIG_T::in_height) { 
      pX = 0;
      pY = pY+1;
      pPass = false;
    }
    if(pX == lShiftX && pY > lShiftY-1) pPass = true;
}
*/
template<class data_T, class res_T, typename CONFIG_T>
void pool_2d_large_stream(
		      hls::stream<data_T> data[CONFIG_T::n_chan],
		      hls::stream<res_T>  res [CONFIG_T::n_filt]) { 

    const static int lShiftX = CONFIG_T::filt_width-CONFIG_T::pad_left-1;
    const static int lShiftY = CONFIG_T::filt_height-CONFIG_T::pad_top-1;

    static data_T layer_in_row[CONFIG_T::in_width+CONFIG_T::pad_left+CONFIG_T::pad_right][CONFIG_T::filt_height][CONFIG_T::n_chan];
    #pragma HLS ARRAY_RESHAPE variable=layer_in_row complete dim=3

    static data_T layer_in[CONFIG_T::filt_height*CONFIG_T::filt_width*CONFIG_T::n_chan];
    //#pragma HLS ARRAY_RESHAPE variable=layer_in block factor=CONFIG_T::n_chan dim=0
    #pragma HLS ARRAY_RESHAPE variable=layer_in complete dim=0

    static unsigned pX=0;
    static unsigned pY=0;
    static bool pPass = false;
    if(pX == 0 && pY == 0) pPass = false;
    for(int i0 = 0; i0 < CONFIG_T::n_chan; i0++) { 
       #pragma HLS UNROLL
       layer_in_row[pX+CONFIG_T::pad_left][(pY+CONFIG_T::pad_top) % CONFIG_T::filt_height][i0] =  data[i0].read();
    }
    //Processs image
    if(pX == 0) nnet::reset_down_2dXNew<data_T,data_T,CONFIG_T>(pX,layer_in_row,layer_in); //check stride
    if((pX+1) % CONFIG_T::stride_width == 0 && (pY+1) % CONFIG_T::stride_height == 0 && pPass) { 
      for(unsigned i0 = 0; i0 < CONFIG_T::n_filt; i0++) { 
        #pragma HLS UNROLL
	data_T pool[CONFIG_T::pool_height * CONFIG_T::pool_width];
        #pragma HLS ARRAY_RESHAPE variable=pool complete dim=0
        for(unsigned i1 = 0; i1 < CONFIG_T::pool_height*CONFIG_T::pool_width; i1++) { 
         #pragma HLS UNROLL
	  pool[i1] = layer_in[i1*CONFIG_T::n_filt+i0];
	}
	res[i0].write(pool_op<data_T, CONFIG_T::pool_height*CONFIG_T::pool_width, CONFIG_T::pool_op>(pool));
      }
      nnet::shift_right_stride_2dNew<data_T,data_T,CONFIG_T>(pX,pY,layer_in_row,layer_in);//check stride
    }    
    pX = pX+1;
    if(pX == CONFIG_T::in_height) { 
      pX = 0;
      pY = pY+1;
      pPass = false;
    }
    if(pX == lShiftX && pY > lShiftY-1) pPass = true;
}


template<class data_T, class res_T, typename CONFIG_T>
void pool_2d_large_stream_funnel(bool iReset,
				 hls::stream<data_T> data[CONFIG_T::n_chan],
				 hls::stream<res_T>  res [CONFIG_T::n_filt]) { 

    const static int lShiftX  = CONFIG_T::filt_width-CONFIG_T::pad_left-1;
    const static int lShiftY  = CONFIG_T::filt_height-CONFIG_T::pad_top-1;  
    const int        rufactor = CONFIG_T::reuse_factor;
    const int    block_factor = CONFIG_T::n_chan/CONFIG_T::reuse_factor;
    //const data_T  avg         = 1./CONFIG_T::filt_width/CONFIG_T::filt_height;
    const int  avg            = CONFIG_T::filt_width*CONFIG_T::filt_height;

    static data_T layer_in[CONFIG_T::n_chan];
    #pragma HLS ARRAY_RESHAPE variable=layer_in block factor=block_factor

    static int pX=0; 
    static int pY=0;
    if(iReset) { 
      pX = 0; 
      pY = 0;
      for(unsigned i0 = 0; i0 < CONFIG_T::n_chan; i0++) { 
	#pragma HLS UNROLL
	layer_in[i0] = 0; 
      }
    }
    static bool pPass = false;
    if(pX == 0 && pY == 0) pPass = false;

    for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 
        for (int im = 0; im < block_factor; im++) {
         int iblock = ir+rufactor*im;
         data_T value1 = data[iblock].read();
         //data_T value2 = layer_in[iblock];
         //pool_op<data_T, CONFIG_T::pool_height*CONFIG_T::pool_width, CONFIG_T::pool_op>(pool);
         //if(value1 > value2) layer_in[iblock] = value1;
	 layer_in[iblock] += value1;
        }
    }
    if((pX+1) % CONFIG_T::stride_width == 0 && (pY+1) % CONFIG_T::stride_height == 0 && pPass) { 
     for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 
        for (int im = 0; im < block_factor; im++) {
	 int iblock = ir*block_factor+im;
         res[iblock].write(layer_in[iblock]/avg);
        }
     }
    }
    pX = pX+1;
    if(pX == CONFIG_T::in_height) { 
      pX = 0;
      pY = pY+1;
      pPass = false;
    }
    if(pX == lShiftX && pY > lShiftY-1) pPass = true;
}

template<class data_T, typename CONFIG_T>
void maxpool2d_filt_cl(data_T data[CONFIG_T::pool_height * CONFIG_T::pool_width * CONFIG_T::n_filt],
		       data_T res[CONFIG_T::n_filt]){
  data_T pMax[CONFIG_T::n_filt];
  #pragma HLS ARRAY_RESHAPE variable=pMax complete dim=0
  for(unsigned i0 = 0; i0 <  CONFIG_T::n_filt; i0++) {
    #pragma HLS UNROLL
    pMax[i0] = data[i0];
  }
  for(unsigned i0 = 0; i0 < CONFIG_T::n_filt; i0++) { 
    //#pragma HLS PIPELINE II=1
    #pragma HLS UNROLL
    for(unsigned i1 = 1; i1 < (CONFIG_T::pool_height*CONFIG_T::pool_width); i1++) { 
     #pragma HLS UNROLL
     data_T pTmp = data[(i1*CONFIG_T::n_filt)+i0];
     if(pMax[i0] < pTmp) pMax[i0] = pTmp;
    }
  }
  for(unsigned i0 = 0; i0 <  CONFIG_T::n_filt; i0++) {
    #pragma HLS UNROLL
    res[i0] = pMax[i0];
  }
}
template<typename CONFIG_T>
constexpr int pool_op_limit(){
  return (CONFIG_T::out_height * CONFIG_T::out_width) * CONFIG_T::n_filt / CONFIG_T::reuse;
}

template<class data_T, typename CONFIG_T>
void pooling2d_cl(data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_filt],
               data_T res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt]){

  // TODO partition the arrays according to the reuse factor
  const int limit = pool_op_limit<CONFIG_T>();
  #pragma HLS ALLOCATION instances=pool_op limit=limit function
  // Add any necessary padding
  unsigned padded_height = CONFIG_T::in_height + CONFIG_T::pad_top + CONFIG_T::pad_bottom;
  unsigned padded_width = CONFIG_T::in_width + CONFIG_T::pad_left + CONFIG_T::pad_right;
  if (CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0 && CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0) {
    padded_height -= padded_height - (padded_height / CONFIG_T::stride_height * CONFIG_T::stride_height);
    padded_width -= padded_width - (padded_width / CONFIG_T::stride_width * CONFIG_T::stride_width);
  }

  for(int ff = 0; ff < CONFIG_T::n_filt; ff++){
	  // Loop over input image y in steps of stride
	  for(int ii = 0; ii < padded_height; ii += CONFIG_T::stride_height){
		  // Loop over input image x in steps of stride
		  for(int jj = 0; jj < padded_width; jj += CONFIG_T::stride_width){
			  data_T pool[CONFIG_T::pool_height * CONFIG_T::pool_width];
        // Keep track of number of pixels in image vs padding region
        unsigned img_overlap = 0;
			  // Loop over pool window y
			  for(int kk = 0; kk < CONFIG_T::stride_height; kk++){
				  // Loop over pool window x
				  for(int ll = 0; ll < CONFIG_T::stride_width; ll++){
            if(ii+kk < CONFIG_T::pad_top || ii+kk >= (padded_height - CONFIG_T::pad_bottom) || jj+ll < CONFIG_T::pad_left || jj+ll >= (padded_width - CONFIG_T::pad_right)){
              // Add padding
              pool[kk * CONFIG_T::stride_width + ll] = pad_val<data_T, CONFIG_T::pool_op>();
            }else{
  					  pool[kk * CONFIG_T::stride_width + ll] = data[(ii + kk) * CONFIG_T::in_width * CONFIG_T::n_filt + (jj + ll) * CONFIG_T::n_filt + ff];
              img_overlap++;
            }
				  }
			  }
			  // do the pooling
        // TODO in the case of average pooling, need to reduce height * width to area of pool window
        // not overlapping padding region
			  res[(ii/CONFIG_T::stride_height) * CONFIG_T::out_width * CONFIG_T::n_filt + (jj/CONFIG_T::stride_width)* CONFIG_T::n_filt + ff] =
					  pool_op<data_T, CONFIG_T::pool_height*CONFIG_T::pool_width, CONFIG_T::pool_op>(pool);
        // If the pool op is Average, the zero-padding needs to be removed from the results
        if(CONFIG_T::pool_op == Average){
          data_T rescale = CONFIG_T::pool_height * CONFIG_T::pool_width / img_overlap;
          res[(ii/CONFIG_T::stride_height) * CONFIG_T::out_width * CONFIG_T::n_filt + (jj/CONFIG_T::stride_width)* CONFIG_T::n_filt + ff] *= rescale;
        }
		  }
	  }
  }
}

template<class data_T, typename CONFIG_T>
void pooling2d_cf(data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_filt],
               data_T res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt]){

  // TODO partition the arrays according to the reuse factor
  const int limit = pool_op_limit<CONFIG_T>();
  #pragma HLS ALLOCATION instances=pool_op limit=limit function
  // Add any necessary padding
  unsigned padded_height = CONFIG_T::in_height + CONFIG_T::pad_top + CONFIG_T::pad_bottom;
  unsigned padded_width = CONFIG_T::in_width + CONFIG_T::pad_left + CONFIG_T::pad_right;
  if (CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0 && CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0) {
    padded_height -= padded_height - (padded_height / CONFIG_T::stride_height * CONFIG_T::stride_height);
    padded_width -= padded_width - (padded_width / CONFIG_T::stride_width * CONFIG_T::stride_width);
  }

  for(int ff = 0; ff < CONFIG_T::n_filt; ff++){
	  // Loop over input image y in steps of stride
	  for(int ii = 0; ii < padded_height; ii += CONFIG_T::stride_height){
		  // Loop over input image x in steps of stride
		  for(int jj = 0; jj < padded_width; jj += CONFIG_T::stride_width){
			  data_T pool[CONFIG_T::pool_height * CONFIG_T::pool_width];
        // Keep track of number of pixels in image vs padding region
        unsigned img_overlap = 0;
			  // Loop over pool window y
			  for(int kk = 0; kk < CONFIG_T::stride_height; kk++){
				  // Loop over pool window x
				  for(int ll = 0; ll < CONFIG_T::stride_width; ll++){
            if(ii+kk < CONFIG_T::pad_top || ii+kk >= (padded_height - CONFIG_T::pad_bottom) || jj+ll < CONFIG_T::pad_left || jj+ll >= (padded_width - CONFIG_T::pad_right)){
              // Add padding
              pool[kk * CONFIG_T::stride_width + ll] = pad_val<data_T, CONFIG_T::pool_op>();
            }else{
  					  pool[kk * CONFIG_T::stride_width + ll] = data[(ii + kk) * CONFIG_T::in_width + ff * CONFIG_T::in_width*CONFIG_T::in_height + ll + jj];
              img_overlap++;
            }
				  }
			  }
			  // do the pooling
        // TODO in the case of average pooling, need to reduce height * width to area of pool window
        // not overlapping padding region
			  res[(ii/CONFIG_T::stride_height) * CONFIG_T::out_width + (jj/CONFIG_T::stride_width) + ff* CONFIG_T::out_height* CONFIG_T::out_width] =
					  pool_op<data_T, CONFIG_T::pool_height*CONFIG_T::pool_width, CONFIG_T::pool_op>(pool);
        // If the pool op is Average, the zero-padding needs to be removed from the results
        if(CONFIG_T::pool_op == Average){
          data_T rescale = CONFIG_T::pool_height * CONFIG_T::pool_width / img_overlap;
          res[(ii/CONFIG_T::stride_height) * CONFIG_T::out_width + (jj/CONFIG_T::stride_width) + ff* CONFIG_T::out_height* CONFIG_T::out_width] *= rescale;
        }
		  }
	  }
  }
}

}

#endif
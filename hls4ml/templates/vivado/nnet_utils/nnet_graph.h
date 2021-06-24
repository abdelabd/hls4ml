#ifndef NNET_GRAPH_H_
#define NNET_GRAPH_H_

#include "nnet_common.h"
#include "nnet_merge.h"
#include "nnet_dense.h"
#include "nnet_dense_resource.h"
#include "nnet_activation.h"
#include "nnet_array.h"
#include "hls_stream.h"
#include <string>
#include <sstream>
#include <math.h>

namespace nnet {
  
  struct graph_config
  {
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    
    // Layer Sizes
    static const unsigned n_node = 10;
    static const unsigned n_edge = 20;
    static const unsigned n_batch = 20;
    static const unsigned n_in = 7;
    static const unsigned n_hidden = 4;
    static const unsigned n_out = 4;
    
    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned io_stream = false;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
  };

  template<class data_T, class res_T, typename CONFIG_T>
    void dense_mult_1lyr(
			 data_T data[CONFIG_T::dense_config1::n_in],
			 res_T res[CONFIG_T::dense_config1::n_out],
			 typename CONFIG_T::dense_config1::weight_t weights0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			 typename CONFIG_T::dense_config1::bias_t   biases0[CONFIG_T::dense_config1::n_out])
  {
    nnet::dense_resource_basic<data_T, res_T, typename CONFIG_T::dense_config1>(data, res, weights0, biases0);
  }

  template<class data_T, class res_T, typename CONFIG_T>
    void dense_mult_2lyr(
			 data_T data[CONFIG_T::dense_config1::n_in],
			 res_T res[CONFIG_T::dense_config2::n_out],
			 typename CONFIG_T::dense_config1::weight_t weights0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			 typename CONFIG_T::dense_config1::bias_t   biases0[CONFIG_T::dense_config1::n_out],
			 typename CONFIG_T::dense_config2::weight_t weights1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
			 typename CONFIG_T::dense_config2::bias_t   biases1[CONFIG_T::dense_config2::n_out])
  {
    data_T data0_logits[CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=data0_logits complete dim=0
    nnet::dense_resource_basic<data_T, data_T, typename CONFIG_T::dense_config1>(data, data0_logits, weights0, biases0);
    data_T data0[CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=data0 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config1>(data0_logits, data0);

    nnet::dense_resource_basic<data_T, res_T, typename CONFIG_T::dense_config2>(data0, res, weights1, biases1);
  }

  template<class data_T, class res_T, typename CONFIG_T>
    void dense_mult_3lyr(
			 data_T data[CONFIG_T::dense_config1::n_in],
			 res_T res[CONFIG_T::dense_config3::n_out],
			 typename CONFIG_T::dense_config1::weight_t weights0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			 typename CONFIG_T::dense_config1::bias_t   biases0[CONFIG_T::dense_config1::n_out],
			 typename CONFIG_T::dense_config2::weight_t weights1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
			 typename CONFIG_T::dense_config2::bias_t   biases1[CONFIG_T::dense_config2::n_out],
			 typename CONFIG_T::dense_config3::weight_t weights2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
			 typename CONFIG_T::dense_config3::bias_t   biases2[CONFIG_T::dense_config3::n_out])
  {
    data_T data0_logits[CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=data0_logits complete dim=0
    nnet::dense_resource_basic<data_T, data_T, typename CONFIG_T::dense_config1>(data, data0_logits, weights0, biases0);
    data_T data0[CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=data0 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config1>(data0_logits, data0);

    data_T data1_logits[CONFIG_T::dense_config2::n_out];
    #pragma HLS ARRAY_PARTITION variable=data1_logits complete dim=0
    nnet::dense_resource_basic<data_T, data_T, typename CONFIG_T::dense_config2>(data0, data1_logits, weights1, biases1);
    data_T data1[CONFIG_T::dense_config2::n_out];
    #pragma HLS ARRAY_PARTITION variable=data1 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config2>(data1_logits, data1);

    nnet::dense_resource_basic<data_T, res_T, typename CONFIG_T::dense_config3>(data1, res, weights2, biases2);
  }

  template<class data_T, class res_T, typename CONFIG_T>
    void dense_mult_4lyr(
			 data_T data[CONFIG_T::dense_config1::n_in],
			 res_T res[CONFIG_T::dense_config4::n_out],
			 typename CONFIG_T::dense_config1::weight_t weights0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			 typename CONFIG_T::dense_config1::bias_t   biases0[CONFIG_T::dense_config1::n_out],
			 typename CONFIG_T::dense_config2::weight_t weights1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
			 typename CONFIG_T::dense_config2::bias_t   biases1[CONFIG_T::dense_config2::n_out],
			 typename CONFIG_T::dense_config3::weight_t weights2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
			 typename CONFIG_T::dense_config3::bias_t   biases2[CONFIG_T::dense_config3::n_out],
			 typename CONFIG_T::dense_config4::weight_t weights3[CONFIG_T::dense_config4::n_in*CONFIG_T::dense_config4::n_out],
			 typename CONFIG_T::dense_config4::bias_t   biases3[CONFIG_T::dense_config4::n_out])
  {
    data_T data0_logits[CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=data0_logits complete dim=0
    nnet::dense_resource_basic<data_T, data_T, typename CONFIG_T::dense_config1>(data, data0_logits, weights0, biases0);
    data_T data0[CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=data0 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config1>(data0_logits, data0);

    data_T data1_logits[CONFIG_T::dense_config2::n_out];
    #pragma HLS ARRAY_PARTITION variable=data1_logits complete dim=0
    nnet::dense_resource_basic<data_T, data_T, typename CONFIG_T::dense_config2>(data0, data1_logits, weights1, biases1);
    data_T data1[CONFIG_T::dense_config2::n_out];
    #pragma HLS ARRAY_PARTITION variable=data1 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config2>(data1_logits, data1);

    data_T data2_logits[CONFIG_T::dense_config3::n_out];
    #pragma HLS ARRAY_PARTITION variable=data2_logits complete dim=0
    nnet::dense_resource_basic<data_T, data_T, typename CONFIG_T::dense_config3>(data1, data2_logits, weights2, biases2);
    data_T data2[CONFIG_T::dense_config3::n_out];
    #pragma HLS ARRAY_PARTITION variable=data2 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config3>(data2_logits, data2);

    nnet::dense_resource_basic<data_T, res_T, typename CONFIG_T::dense_config4>(data2, res, weights3, biases3);
  }

  template<class data_T, class index_T, class res_T, typename CONFIG_T>
    void IN_edge_module(
            data_T    Rn_1D[CONFIG_T::n_node*CONFIG_T::n_features],
			data_T    Re_1D[CONFIG_T::n_edge*CONFIG_T::e_features],
			index_T   edge_index_1D[CONFIG_T::n_edge*2],
			res_T     L_1D[CONFIG_T::n_edge*CONFIG_T::n_out],
			res_T     Q_1D[CONFIG_T::n_node*CONFIG_T::n_out],
			typename CONFIG_T::dense_config1::weight_t  core_edge_w0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			typename CONFIG_T::dense_config1::bias_t    core_edge_b0[CONFIG_T::dense_config1::n_out],
			typename CONFIG_T::dense_config2::weight_t  core_edge_w1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
			typename CONFIG_T::dense_config2::bias_t    core_edge_b1[CONFIG_T::dense_config2::n_out],
			typename CONFIG_T::dense_config3::weight_t  core_edge_w2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
			typename CONFIG_T::dense_config3::bias_t    core_edge_b2[CONFIG_T::dense_config3::n_out],
			typename CONFIG_T::dense_config4::weight_t  core_edge_w3[CONFIG_T::dense_config4::n_in*CONFIG_T::dense_config4::n_out],
			typename CONFIG_T::dense_config4::bias_t    core_edge_b3[CONFIG_T::dense_config4::n_out])

  {
    //input vectors --> input arrays
    // 1. Rn
    data_T Rn[CONFIG_T::n_node][CONFIG_T::n_features];
    nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::Rn_config>(Rn_1D, Rn);

    // 2. Re
    data_T Re[CONFIG_T::n_edge][CONFIG_T::e_features];
    nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::Re_config>(Re_1D, Re);

    // 3. edge_index
    index_T edge_index[2][CONFIG_T::n_edge];
    nnet::vec_to_mat<index_T, index_T, typename CONFIG_T::edge_index_config>(edge_index_1D, edge_index);

    //output arrays
    // 1.L
    res_T L[CONFIG_T::n_edge][CONFIG_T::n_out];

    // 2. Q
    res_T Q[CONFIG_T::n_node][CONFIG_T::n_out];
    for(int i = 0; i < CONFIG_T::n_node; i++){
      for(int j = 0; j < CONFIG_T::n_out; j++){
	    Q[i][j] = 0;
      }
    }

    if(CONFIG_T::io_stream){
      #pragma HLS STREAM variable=edge_index
    }

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    IN_edge_loop: for(int i = 0; i < CONFIG_T::n_edge; i++) {
      #pragma HLS UNROLL
      index_T r = edge_index[1][i]; // 'x_i'
      index_T s = edge_index[0][i]; // 'x_j'

      data_T l_logits[2*CONFIG_T::n_features];
      #pragma HLS ARRAY_PARTITION variable=l_logits complete dim=0
      nnet::concatenate1d<data_T, data_T, data_T, typename CONFIG_T::merge_config1>(Rn[r], Rn[s], l_logits);
      data_T l[CONFIG_T::e_features + 2*CONFIG_T::n_features];
      #pragma HLS ARRAY_PARTITION variable=l complete dim=0
      nnet::concatenate1d<data_T, data_T, data_T, typename CONFIG_T::merge_config2>(l_logits, Re[i], l);

      if(CONFIG_T::activate_final){
	data_T L_logits[CONFIG_T::n_out];
        #pragma HLS ARRAY_PARTITION variable=L_logits complete dim=0
        if(CONFIG_T::n_layers == 1){
	  nnet::dense_mult_1lyr<data_T, data_T, CONFIG_T>(l, L_logits, core_edge_w0, core_edge_b0);
	  nnet::relu<data_T, res_T, typename CONFIG_T::relu_config1>(L_logits, L[i]);
        }else if(CONFIG_T::n_layers == 2){
	  nnet::dense_mult_2lyr<data_T, data_T, CONFIG_T>(l, L_logits, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1);
	  nnet::relu<data_T, res_T, typename CONFIG_T::relu_config2>(L_logits, L[i]);
	}else if(CONFIG_T::n_layers == 3){
	  nnet::dense_mult_3lyr<data_T, data_T, CONFIG_T>(l, L_logits, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2);
	  nnet::relu<data_T, res_T, typename CONFIG_T::relu_config3>(L_logits, L[i]);
	}else if(CONFIG_T::n_layers == 4){
	  nnet::dense_mult_4lyr<data_T, data_T, CONFIG_T>(l, L_logits, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
	  nnet::relu<data_T, res_T, typename CONFIG_T::relu_config4>(L_logits, L[i]);
	}
      }else{
        if(CONFIG_T::n_layers == 1){
	  nnet::dense_mult_1lyr<data_T, data_T, CONFIG_T>(l, L[i], core_edge_w0, core_edge_b0);
        }else if(CONFIG_T::n_layers == 2){
	  nnet::dense_mult_2lyr<data_T, data_T, CONFIG_T>(l, L[i], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1);
        }else if(CONFIG_T::n_layers == 3){
	  nnet::dense_mult_3lyr<data_T, data_T, CONFIG_T>(l, L[i], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2);
        }else if(CONFIG_T::n_layers == 4){
	  nnet::dense_mult_4lyr<data_T, res_T, CONFIG_T>(l, L[i], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
        }
      }

      for(int j = 0; j < CONFIG_T::n_out; j++){
        #pragma HLS UNROLL
	Q[r][j] += L[i][j];
      }
    }

    //output arrays --> output vectors
    // 1. L_1D
    nnet::mat_to_vec<res_T, res_T, typename CONFIG_T::L_config>(L, L_1D);

    // 2. Q_1D
    nnet::mat_to_vec<res_T, res_T, typename CONFIG_T::Q_config>(Q, Q_1D);
  }

  template<class data_T, class res_T, typename CONFIG_T>
    void IN_node_module(
			data_T    Rn_1D[CONFIG_T::n_node*CONFIG_T::n_features],
			data_T    Q_1D[CONFIG_T::n_node*CONFIG_T::e_features],
			res_T     P_1D[CONFIG_T::n_node*CONFIG_T::n_out],
			typename CONFIG_T::dense_config1::weight_t  core_node_w0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			typename CONFIG_T::dense_config1::bias_t    core_node_b0[CONFIG_T::dense_config1::n_out],
			typename CONFIG_T::dense_config2::weight_t  core_node_w1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
			typename CONFIG_T::dense_config2::bias_t    core_node_b1[CONFIG_T::dense_config2::n_out],
			typename CONFIG_T::dense_config3::weight_t  core_node_w2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
			typename CONFIG_T::dense_config3::bias_t    core_node_b2[CONFIG_T::dense_config3::n_out],
			typename CONFIG_T::dense_config4::weight_t  core_node_w3[CONFIG_T::dense_config4::n_in*CONFIG_T::dense_config4::n_out],
			typename CONFIG_T::dense_config4::bias_t    core_node_b3[CONFIG_T::dense_config4::n_out])
  {
    //input vectors --> input arrays
    //1. Rn
    data_T Rn[CONFIG_T::n_node][CONFIG_T::n_features];
    nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::Rn_config>(Rn_1D, Rn);

    //2. Q
    data_T Q[CONFIG_T::n_node][CONFIG_T::e_features];
    nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::Q_config>(Q_1D, Q);

    //output array
    // 1. P
    res_T P[CONFIG_T::n_node][CONFIG_T::n_out];

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    IN_node_loop: for(int i = 0; i < CONFIG_T::n_node; i++){
      #pragma HLS UNROLL
      data_T p[CONFIG_T::e_features + CONFIG_T::n_features];
      #pragma HLS ARRAY_PARTITION variable=p complete dim=0
      nnet::concatenate1d<data_T, data_T, data_T, typename CONFIG_T::merge_config1>(Rn[i], Q[i], p);

      if(CONFIG_T::activate_final){
	data_T P_logits[CONFIG_T::n_out];
	#pragma HLS ARRAY_PARTITION variable=P_logits complete dim=0
	if(CONFIG_T::n_layers == 1){
	  nnet::dense_mult_1lyr<data_T, data_T, CONFIG_T>(p, P_logits, core_node_w0, core_node_b0);
	  nnet::relu<data_T, res_T, typename CONFIG_T::relu_config1>(P_logits, P[i]);
	}else if(CONFIG_T::n_layers == 2){
	  nnet::dense_mult_2lyr<data_T, data_T, CONFIG_T>(p, P_logits, core_node_w0, core_node_b0, core_node_w1, core_node_b1);
	  nnet::relu<data_T, res_T, typename CONFIG_T::relu_config2>(P_logits, P[i]);
	}else if(CONFIG_T::n_layers == 3){
	  nnet::dense_mult_3lyr<data_T, data_T, CONFIG_T>(p, P_logits, core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2);
	  nnet::relu<data_T, res_T, typename CONFIG_T::relu_config3>(P_logits, P[i]);
	}else if(CONFIG_T::n_layers == 4){
	  nnet::dense_mult_4lyr<data_T, data_T, CONFIG_T>(p, P_logits, core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2, core_node_w3, core_node_b3);
	  nnet::relu<data_T, res_T, typename CONFIG_T::relu_config4>(P_logits, P[i]);
	}
      }else{
        if(CONFIG_T::n_layers == 1){
	  nnet::dense_mult_1lyr<data_T, res_T, CONFIG_T>(p, P[i], core_node_w0, core_node_b0);
        }else if(CONFIG_T::n_layers == 2){
	  nnet::dense_mult_2lyr<data_T, res_T, CONFIG_T>(p, P[i], core_node_w0, core_node_b0, core_node_w1, core_node_b1);
        }else if(CONFIG_T::n_layers == 3){
	  nnet::dense_mult_3lyr<data_T, res_T, CONFIG_T>(p, P[i], core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2);
        }else if(CONFIG_T::n_layers == 4){
	  nnet::dense_mult_4lyr<data_T, res_T, CONFIG_T>(p, P[i], core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2, core_node_w3, core_node_b3);
        }
      }
    }

    // output array --> output vector
    nnet::mat_to_vec<res_T, res_T, typename CONFIG_T::P_config>(P, P_1D);
  }

  template<class data_T, class res_T, typename CONFIG_T>
    void graph_independent(
			   data_T    X[CONFIG_T::dense_config1::n_batch][CONFIG_T::dense_config1::n_in],
			   res_T     R[CONFIG_T::dense_config2::n_batch][CONFIG_T::dense_config2::n_out],
			   typename CONFIG_T::dense_config1::weight_t  w0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			   typename CONFIG_T::dense_config1::bias_t    b0[CONFIG_T::dense_config1::n_out],
			   typename CONFIG_T::dense_config2::weight_t  w1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
			   typename CONFIG_T::dense_config2::bias_t    b1[CONFIG_T::dense_config2::n_out])
  {
    if(CONFIG_T::io_stream){
      #pragma HLS STREAM variable=X
    }
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    data_T R0_logits[CONFIG_T::dense_config1::n_batch][CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=R0_logits complete dim=0
    nnet::dense_batch<data_T, data_T, typename CONFIG_T::dense_config1>(X, R0_logits, w0, b0);
    data_T R0[CONFIG_T::relu_config1::n_batch][CONFIG_T::relu_config1::n_in];
    #pragma HLS ARRAY_PARTITION variable=R0 complete dim=0
    nnet::relu_batch<data_T, data_T, typename CONFIG_T::relu_config1>(R0_logits, R0);

    if(CONFIG_T::activate_final){
        data_T R_logits[CONFIG_T::dense_config2::n_batch][CONFIG_T::dense_config2::n_out];
        #pragma HLS ARRAY_PARTITION variable=R_logits complete dim=0
        nnet::dense_batch<data_T, data_T, typename CONFIG_T::dense_config2>(R0, R_logits, w1, b1);
        nnet::relu_batch<data_T, res_T, typename CONFIG_T::relu_config2>(R_logits, R);
    }else{
      nnet::dense_batch<data_T, data_T, typename CONFIG_T::dense_config2>(R0, R, w1, b1);
    }
  }

}

#endif

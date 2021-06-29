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
    static const unsigned n_out = 4;
    static const unsigned n_layers = 3;
    static const unsigned n_features = 3;
    static const unsigned e_features = 4;
    
    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned io_stream = false;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;

    static const bool activate_final = false;
  };

  struct aggregate_config
  {
     static const unsigned n_node = 10;
     static const unsigned n_edge = 20;
     static const unsigned edge_dim = 4;
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
    void dense_mult_3lyr_with_save(
			 data_T data[CONFIG_T::dense_config1::n_in],
			 res_T res[CONFIG_T::dense_config3::n_out],
			 res_T fc1_output[CONFIG_T::dense_config1::n_out],
			 res_T relu1_output[CONFIG_T::dense_config1::n_out],
			 res_T fc2_output[CONFIG_T::dense_config2::n_out],
			 res_T relu2_output[CONFIG_T::dense_config2::n_out],
			 res_T fc3_output[CONFIG_T::dense_config3::n_out],
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
    nnet::dense_resource_basic<data_T, data_T, typename CONFIG_T::dense_config1>(data, fc1_output, weights0, biases0);
    data_T data0[CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=data0 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config1>(data0_logits, data0);
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config1>(data0_logits, relu1_output);

    data_T data1_logits[CONFIG_T::dense_config2::n_out];
    #pragma HLS ARRAY_PARTITION variable=data1_logits complete dim=0
    nnet::dense_resource_basic<data_T, data_T, typename CONFIG_T::dense_config2>(data0, data1_logits, weights1, biases1);
    nnet::dense_resource_basic<data_T, data_T, typename CONFIG_T::dense_config2>(data0, fc2_output, weights1, biases1);
    data_T data1[CONFIG_T::dense_config2::n_out];
    #pragma HLS ARRAY_PARTITION variable=data1 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config2>(data1_logits, data1);
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config2>(data1_logits, relu2_output);

    nnet::dense_resource_basic<data_T, res_T, typename CONFIG_T::dense_config3>(data1, res, weights2, biases2);
    nnet::dense_resource_basic<data_T, res_T, typename CONFIG_T::dense_config3>(data1, fc3_output, weights2, biases2);
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
    void aggregate_add(
            data_T    edge_attr[CONFIG_T::n_edge][CONFIG_T::edge_dim],
            index_T   edge_index[CONFIG_T::n_edge][2],
            res_T     edge_attr_aggr[CONFIG_T::n_node][CONFIG_T::edge_dim])
    {
      for(int i=0; i<CONFIG_T::n_edge; i++){

        index_T r = edge_index[i][1]; // 'x_i'
        for(int j=0; j<CONFIG_T::edge_dim; j++){
          #pragma HLS UNROLL
          edge_attr_aggr[r][j] += edge_attr[i][j];
        }
      }
    }

  template<class data_T, class res_T, typename CONFIG_T>
    void aggregate_single_edge_add(
          data_T    edge_attr_single[CONFIG_T::edge_dim],
          res_T     edge_attr_aggr[CONFIG_T::edge_dim]
    )
    {
      for(int j=0; j<CONFIG_T::edge_dim; j++){
        #pragma HLS UNROLL
        edge_attr_aggr[j] += edge_attr_single[j];
      }
    }

  template<class data_T, class res_T, typename CONFIG_T>
    void aggregate_single_edge_max( //maybe max-pooling instead?
          data_T    edge_attr_single[CONFIG_T::edge_dim],
          res_T     edge_attr_aggr[CONFIG_T::edge_dim]
    )
    {
      for(int j=0; j<CONFIG_T::edge_dim; j++){
        #pragma HLS UNROLL
        edge_attr_aggr[j] = edge_attr_single[j] > edge_attr_aggr[j] ? edge_attr_single[j] : edge_attr_aggr[j];
        //if(edge_attr_aggr[j] < edge_attr_single[j]){
          //edge_attr_aggr[j] = edge_attr_single[j];
        //}
      }
    }

  template<class data_T, class index_T, class res_T, typename CONFIG_T>
    void EdgeBlock(
            data_T    node_attr_1D[CONFIG_T::n_node*CONFIG_T::node_dim],
			data_T    edge_attr_1D[CONFIG_T::n_edge*CONFIG_T::edge_dim],
			index_T   edge_index_1D[CONFIG_T::n_edge*2],
			res_T     edge_update_1D[CONFIG_T::n_edge*CONFIG_T::out_dim],
			res_T     edge_update_aggr_1D[CONFIG_T::n_node*CONFIG_T::out_dim],
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
    // 1. node_attr
    data_T node_attr[CONFIG_T::n_node][CONFIG_T::node_dim];
    nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::node_attr_config>(node_attr_1D, node_attr);

    // 2. edge_attr
    data_T edge_attr[CONFIG_T::n_edge][CONFIG_T::edge_dim];
    nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::edge_attr_config>(edge_attr_1D, edge_attr);

    // 3. edge_index
    index_T edge_index[CONFIG_T::n_edge][2];
    nnet::vec_to_mat<index_T, index_T, typename CONFIG_T::edge_index_config>(edge_index_1D, edge_index);

    //output arrays
    // 1. edge_update
    res_T edge_update[CONFIG_T::n_edge][CONFIG_T::out_dim];

    // 2. edge_update_aggr
    res_T edge_update_aggr[CONFIG_T::n_node][CONFIG_T::out_dim];
    for(int i = 0; i < CONFIG_T::n_node; i++){
      for(int j = 0; j < CONFIG_T::out_dim; j++){
	    edge_update_aggr[i][j] = 0;
      }
    }

    // intermediate: edge counter, only useful if aggr==mean
    index_T num_edge_per_node[CONFIG_T::n_node];
    if(CONFIG_T::aggr==1){ //if aggregation-method is mean
      for(int i=0; i<CONFIG_T::n_node; i++){
        num_edge_per_node[i] = 0;
      }
    }

    // intermediates: block_inputs and layer_outputs, only useful if save_intermediates==1
    data_T block_inputs[CONFIG_T::n_edge][CONFIG_T::edge_dim+2*CONFIG_T::node_dim];
    data_T fc1_out[CONFIG_T::n_edge][CONFIG_T::dense_config1::n_out];
    data_T relu1_out[CONFIG_T::n_edge][CONFIG_T::dense_config1::n_out];
    data_T fc2_out[CONFIG_T::n_edge][CONFIG_T::dense_config2::n_out];
    data_T relu2_out[CONFIG_T::n_edge][CONFIG_T::dense_config2::n_out];
    data_T fc3_out[CONFIG_T::n_edge][CONFIG_T::dense_config3::n_out];

    if(CONFIG_T::io_stream){
      #pragma HLS STREAM variable=edge_index
    }

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    edge_loop: for(int i = 0; i < CONFIG_T::n_edge; i++) {
      #pragma HLS UNROLL

      index_T s;
      index_T r;
      if(CONFIG_T::flow==0){
        s = edge_index[i][0]; // sender
        r = edge_index[i][1]; // receiver
      }
      else{
        s = edge_index[i][1]; // sender
        r = edge_index[i][0]; // receiver
      }
      if(CONFIG_T::aggr==1){ //if aggregation-method is mean
        num_edge_per_node[r] += 1;
      }

      data_T node_concat[2*CONFIG_T::node_dim];
      #pragma HLS ARRAY_PARTITION variable=l_logits complete dim=0
      nnet::concatenate1d<data_T, data_T, data_T, typename CONFIG_T::merge_config1>(node_attr[r], node_attr[s], node_concat);
      data_T phi_input[CONFIG_T::edge_dim + 2*CONFIG_T::node_dim];
      #pragma HLS ARRAY_PARTITION variable=l complete dim=0
      nnet::concatenate1d<data_T, data_T, data_T, typename CONFIG_T::merge_config2>(node_concat, edge_attr[i], phi_input);

      if(CONFIG_T::save_intermediates==1){
        for(int j=0; j<CONFIG_T::edge_dim+2*CONFIG_T::node_dim; j++){
          block_inputs[i][j] = phi_input[j];
        }
      }

      if(CONFIG_T::activate_final){
	    data_T edge_update_logits[CONFIG_T::out_dim];
        #pragma HLS ARRAY_PARTITION variable=edge_update_logits complete dim=0
        if(CONFIG_T::n_layers == 1){
	      nnet::dense_mult_1lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update_logits, core_edge_w0, core_edge_b0);
	      nnet::relu<data_T, res_T, typename CONFIG_T::relu_config1>(edge_update_logits, edge_update[i]);
        }
        else if(CONFIG_T::n_layers == 2){
	      nnet::dense_mult_2lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update_logits, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1);
	      nnet::relu<data_T, res_T, typename CONFIG_T::relu_config2>(edge_update_logits, edge_update[i]);
	    }
	    else if(CONFIG_T::n_layers == 3){
	      nnet::dense_mult_3lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update_logits, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2);
	      nnet::relu<data_T, res_T, typename CONFIG_T::relu_config3>(edge_update_logits, edge_update[i]);
	    }
	    else if(CONFIG_T::n_layers == 4){
	      nnet::dense_mult_4lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update_logits, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
	      nnet::relu<data_T, res_T, typename CONFIG_T::relu_config4>(edge_update_logits, edge_update[i]);
	    }
      }
      else{
        if(CONFIG_T::n_layers == 1){
	      nnet::dense_mult_1lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update[i], core_edge_w0, core_edge_b0);
          }
        else if(CONFIG_T::n_layers == 2){
	      nnet::dense_mult_2lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update[i], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1);
        }
        else if(CONFIG_T::n_layers == 3){
	      //nnet::dense_mult_3lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update[i], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2);
          nnet::dense_mult_3lyr_with_save<data_T, data_T, CONFIG_T>(phi_input, edge_update[i], fc1_out[i], relu1_out[i], fc2_out[i], relu2_out[i], fc3_out[i], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2);
        }
        else if(CONFIG_T::n_layers == 4){
	      nnet::dense_mult_4lyr<data_T, res_T, CONFIG_T>(phi_input, edge_update[i], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
        }
      }

      if((CONFIG_T::aggr==0)||(CONFIG_T::aggr==1)){ //if aggregation-method is "add" or "mean"
        nnet::aggregate_single_edge_add<res_T, res_T, typename CONFIG_T::aggregation_config1>(edge_update[i], edge_update_aggr[r]);
      }
      else if(CONFIG_T::aggr==2){ //if aggregation-method is "max"
        nnet::aggregate_single_edge_max<res_T, res_T, typename CONFIG_T::aggregation_config1>(edge_update[i], edge_update_aggr[r]);
      }
    }

    if(CONFIG_T::aggr==1){ //if aggregation-method is "mean"
      for(int i=0; i<CONFIG_T::n_node; i++){
        if(num_edge_per_node[i] > 1){
          for (int j=0; j<CONFIG_T::out_dim; j++){
            edge_update_aggr[i][j] = edge_update_aggr[i][j]/num_edge_per_node[i];
          }
        }
      }
    }

    //output arrays --> output vectors
    // 1. edge_update_1D
    nnet::mat_to_vec<res_T, res_T, typename CONFIG_T::edge_update_config>(edge_update, edge_update_1D);

    // 2. edge_update_aggr_1D
    nnet::mat_to_vec<res_T, res_T, typename CONFIG_T::edge_update_aggr_config>(edge_update_aggr, edge_update_aggr_1D);

    //save intermediates, if applicable
    if(CONFIG_T::save_intermediates==1){
      std::ostringstream out_ss;
      std::ostringstream out_aggr_ss;
      std::ostringstream input_ss ;
      std::ostringstream n_edge_ss;
      std::ostringstream fc1_ss;
      std::ostringstream relu1_ss;
      std::ostringstream fc2_ss;
      std::ostringstream relu2_ss;
      std::ostringstream fc3_ss;

      if(CONFIG_T::out_dim==4){
        out_ss << "R1_out.csv";
        out_aggr_ss << "R1_out_aggr.csv";
        input_ss << "R1_block_inputs.csv";
        n_edge_ss << "R1_num_edge_per_node.csv";
        fc1_ss << "R1_fc1_out.csv";
        relu1_ss << "R1_relu1_out.csv";
        fc2_ss << "R1_fc2_out.csv";
        relu2_ss << "R1_relu2_out.csv";
        fc3_ss << "R1_fc3_out.csv";
      }
      else{
        out_ss << "R2_out.csv";
        out_aggr_ss << "R2_out_aggr.csv";
        input_ss << "R2_block_inputs.csv";
        n_edge_ss << "R2_num_edge_per_node.csv";
        fc1_ss << "R2_fc1_out.csv";
        relu1_ss << "R2_relu1_out.csv";
        fc2_ss << "R2_fc2_out.csv";
        relu2_ss << "R2_relu2_out.csv";
        fc3_ss << "R2_fc3_out.csv";
      }

      std::ofstream out_save;
      std::ofstream out_aggr_save;
      std::ofstream input_save;
      std::ofstream fc1_save;
      std::ofstream relu1_save;
      std::ofstream fc2_save;
      std::ofstream relu2_save;
      std::ofstream fc3_save;

      out_save.open(out_ss.str());
      out_aggr_save.open(out_aggr_ss.str());
      input_save.open(input_ss.str());
      fc1_save.open(fc1_ss.str());
      relu1_save.open(relu1_ss.str());
      fc2_save.open(fc2_ss.str());
      relu2_save.open(relu2_ss.str());
      fc3_save.open(fc3_ss.str());

      for(int i=0; i<CONFIG_T::n_edge; i++){
        //save inputs
        for(int j=0; j<CONFIG_T::edge_dim+2*CONFIG_T::node_dim; j++){
          if(j < CONFIG_T::edge_dim+2*CONFIG_T::node_dim-1){
            input_save << block_inputs[i][j] << ",";
          }
          else{
            input_save << block_inputs[i][j] << std::endl;
          }
        }

        //save outputs
        for(int j=0; j<CONFIG_T::out_dim; j++){
          if(j < CONFIG_T::out_dim-1){
            out_save << edge_update[i][j] << ",";
          }
          else{
            out_save << edge_update[i][j] << std::endl;
          }
        }

        //save fc1_out, relu1_out
        for(int j=0; j<CONFIG_T::dense_config1::n_out; j++){
          if(j < CONFIG_T::dense_config1::n_out - 1){
            fc1_save << fc1_out[i][j] << ",";
            relu1_save << relu1_out[i][j] << ",";
          }
          else{
            fc1_save << fc1_out[i][j] << std::endl;
            relu1_save << relu1_out[i][j] << std::endl;
          }
        }

        //save fc2_out, relu2_out
        for(int j=0; j<CONFIG_T::dense_config2::n_out; j++){
          if(j < CONFIG_T::dense_config2::n_out - 1){
            fc2_save << fc2_out[i][j] << ",";
            relu2_save << relu2_out[i][j] << ",";
          }
          else{
            fc2_save << fc2_out[i][j] << std::endl;
            relu2_save << relu2_out[i][j] << std::endl;
          }
        }

        //save fc3_out
        for(int j=0; j<CONFIG_T::dense_config3::n_out; j++){
          if(j < CONFIG_T::dense_config3::n_out - 1){
            fc3_save << fc3_out[i][j] << ",";
          }
          else{
            fc3_save << fc3_out[i][j] << std::endl;
          }
        }

      }
      out_save.close();
      input_save.close();
      fc1_save.close();
      relu1_save.close();
      fc2_save.close();
      relu2_save.close();
      fc3_save.close();

      // save aggregate output
      for(int i=0; i<CONFIG_T::n_node; i++){
        for(int j=0; j<CONFIG_T::out_dim; j++){
          if(j < CONFIG_T::out_dim-1){
            out_aggr_save << edge_update_aggr[i][j] << ",";
          }
          else{
            out_aggr_save << edge_update_aggr[i][j] << std::endl;
          }
        }
      }
      out_aggr_save.close();

      if(CONFIG_T::aggr==1){//if aggregation-method is "mean"
        std::ofstream n_edge_save;
        n_edge_save.open(n_edge_ss.str());
        for(int i=0; i<CONFIG_T::n_node; i++){
          n_edge_save << num_edge_per_node[i];
        }
      }

    }

  }

  template<class data_T, class res_T, typename CONFIG_T>
    void NodeBlock(
			data_T    node_attr_1D[CONFIG_T::n_node*CONFIG_T::node_dim],
			data_T    edge_attr_aggr_1D[CONFIG_T::n_node*CONFIG_T::edge_dim],
			res_T     node_update_1D[CONFIG_T::n_node*CONFIG_T::out_dim],
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
    //1. node_attr
    data_T node_attr[CONFIG_T::n_node][CONFIG_T::node_dim];
    nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::node_attr_config>(node_attr_1D, node_attr);

    //2. edge_attr_aggr
    data_T edge_attr_aggr[CONFIG_T::n_node][CONFIG_T::edge_dim];
    nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::edge_attr_aggr_config>(edge_attr_aggr_1D, edge_attr_aggr);

    //output array
    // 1. node_update
    res_T node_update[CONFIG_T::n_node][CONFIG_T::out_dim];

    //intermediates: block_inputs and layer_outputs, only useful if we're saving intermediates
    data_T block_inputs[CONFIG_T::n_node][CONFIG_T::edge_dim+CONFIG_T::node_dim];
    data_T fc1_out[CONFIG_T::n_edge][CONFIG_T::dense_config1::n_out];
    data_T relu1_out[CONFIG_T::n_edge][CONFIG_T::dense_config1::n_out];
    data_T fc2_out[CONFIG_T::n_edge][CONFIG_T::dense_config2::n_out];
    data_T relu2_out[CONFIG_T::n_edge][CONFIG_T::dense_config2::n_out];
    data_T fc3_out[CONFIG_T::n_edge][CONFIG_T::dense_config3::n_out];

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    node_loop: for(int i = 0; i < CONFIG_T::n_node; i++){
      #pragma HLS UNROLL
      data_T phi_input[CONFIG_T::edge_dim + CONFIG_T::node_dim];
      #pragma HLS ARRAY_PARTITION variable=p complete dim=0
      nnet::concatenate1d<data_T, data_T, data_T, typename CONFIG_T::merge_config1>(node_attr[i], edge_attr_aggr[i], phi_input);

      if(CONFIG_T::save_intermediates==1){
        for(int j=0; j<CONFIG_T::edge_dim+CONFIG_T::node_dim; j++){
          block_inputs[i][j] = phi_input[j];
        }
      }

      if(CONFIG_T::activate_final){
	data_T node_update_logits[CONFIG_T::node_dim];
	#pragma HLS ARRAY_PARTITION variable=node_update_logits complete dim=0
	if(CONFIG_T::n_layers == 1){
	  nnet::dense_mult_1lyr<data_T, data_T, CONFIG_T>(phi_input, node_update_logits, core_node_w0, core_node_b0);
	  nnet::relu<data_T, res_T, typename CONFIG_T::relu_config1>(node_update_logits, node_update[i]);
	}else if(CONFIG_T::n_layers == 2){
	  nnet::dense_mult_2lyr<data_T, data_T, CONFIG_T>(phi_input, node_update_logits, core_node_w0, core_node_b0, core_node_w1, core_node_b1);
	  nnet::relu<data_T, res_T, typename CONFIG_T::relu_config2>(node_update_logits, node_update[i]);
	}else if(CONFIG_T::n_layers == 3){
	  nnet::dense_mult_3lyr<data_T, data_T, CONFIG_T>(phi_input, node_update_logits, core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2);
	  nnet::relu<data_T, res_T, typename CONFIG_T::relu_config3>(node_update_logits, node_update[i]);
	}else if(CONFIG_T::n_layers == 4){
	  nnet::dense_mult_4lyr<data_T, data_T, CONFIG_T>(phi_input, node_update_logits, core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2, core_node_w3, core_node_b3);
	  nnet::relu<data_T, res_T, typename CONFIG_T::relu_config4>(node_update_logits, node_update[i]);
	}
      }else{
        if(CONFIG_T::n_layers == 1){
	  nnet::dense_mult_1lyr<data_T, res_T, CONFIG_T>(phi_input, node_update[i], core_node_w0, core_node_b0);
        }else if(CONFIG_T::n_layers == 2){
	  nnet::dense_mult_2lyr<data_T, res_T, CONFIG_T>(phi_input, node_update[i], core_node_w0, core_node_b0, core_node_w1, core_node_b1);
        }else if(CONFIG_T::n_layers == 3){
	  nnet::dense_mult_3lyr_with_save<data_T, res_T, CONFIG_T>(phi_input, node_update[i], fc1_out[i], relu1_out[i], fc2_out[i], relu2_out[i], fc3_out[i], core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2);
        }else if(CONFIG_T::n_layers == 4){
	  nnet::dense_mult_4lyr<data_T, res_T, CONFIG_T>(phi_input, node_update[i], core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2, core_node_w3, core_node_b3);
        }
      }
    }

    // output array --> output vector
    nnet::mat_to_vec<res_T, res_T, typename CONFIG_T::node_update_config>(node_update, node_update_1D);

    //save intermediates, if applicable
    if(CONFIG_T::save_intermediates==1){
      std::ostringstream out_ss;
      std::ostringstream input_ss;
      std::ostringstream fc1_ss;
      std::ostringstream relu1_ss;
      std::ostringstream fc2_ss;
      std::ostringstream relu2_ss;
      std::ostringstream fc3_ss;

      out_ss << "O_out.csv";
      input_ss << "O_block_inputs.csv";
      fc1_ss << "O_fc1_out.csv";
      relu1_ss << "O_relu1_out.csv";
      fc2_ss << "O_fc2_out.csv";
      relu2_ss << "O_relu2_out.csv";
      fc3_ss << "O_fc3_out.csv";

      std::ofstream out_save;
      std::ofstream input_save;
      std::ofstream fc1_save;
      std::ofstream relu1_save;
      std::ofstream fc2_save;
      std::ofstream relu2_save;
      std::ofstream fc3_save;

      out_save.open(out_ss.str());
      input_save.open(input_ss.str());
      fc1_save.open(fc1_ss.str());
      relu1_save.open(relu1_ss.str());
      fc2_save.open(fc2_ss.str());
      relu2_save.open(relu2_ss.str());
      fc3_save.open(fc3_ss.str());

      for(int i=0; i<CONFIG_T::n_node; i++){

        //save inputs
        for(int j=0; j<CONFIG_T::edge_dim+CONFIG_T::node_dim; j++){
          if(j < CONFIG_T::edge_dim+CONFIG_T::node_dim-1){
            input_save << block_inputs[i][j] << ",";
          }
          else{
            input_save << block_inputs[i][j] << std::endl;
          }
        }

        //save outputs
        for(int j=0; j<CONFIG_T::out_dim; j++){
          if(j < CONFIG_T::out_dim-1){
            out_save << node_update[i][j] << ",";
          }
          else{
            out_save << node_update[i][j] << std::endl;
          }
        }

        //save fc1_out, relu1_out
        for(int j=0; j<CONFIG_T::dense_config1::n_out; j++){
          if(j < CONFIG_T::dense_config1::n_out - 1){
            fc1_save << fc1_out[i][j] << ",";
            relu1_save << relu1_out[i][j] << ",";
          }
          else{
            fc1_save << fc1_out[i][j] << std::endl;
            relu1_save << relu1_out[i][j] << std::endl;
          }
        }

        //save fc2_out, relu2_out
        for(int j=0; j<CONFIG_T::dense_config2::n_out; j++){
          if(j < CONFIG_T::dense_config2::n_out - 1){
            fc2_save << fc2_out[i][j] << ",";
            relu2_save << relu2_out[i][j] << ",";
          }
          else{
            fc2_save << fc2_out[i][j] << std::endl;
            relu2_save << relu2_out[i][j] << std::endl;
          }
        }

        //save fc3_out
        for(int j=0; j<CONFIG_T::dense_config3::n_out; j++){
          if(j < CONFIG_T::dense_config1::n_out - 1){
            fc3_save << fc3_out[i][j] << ",";
          }
          else{
            fc3_save << fc3_out[i][j] << std::endl;
          }
        }

      }
      out_save.close();
      input_save.close();
      fc1_save.close();
      relu1_save.close();
      fc2_save.close();
      relu2_save.close();
      fc3_save.close();
    }

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

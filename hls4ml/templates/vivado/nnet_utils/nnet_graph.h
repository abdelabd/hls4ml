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
      }
    }

  template<class data_T, class res_T, typename CONFIG_T>
    void replace_single_edge(
      data_T edge_attr_single[CONFIG_T::edge_dim],
      res_T  edge_attr_aggr[CONFIG_T::edge_dim]
    )
    {
      for(int j=0; j<CONFIG_T::edge_dim; j++){
        #pragma HLS UNROLL
        edge_attr_aggr[j] = edge_attr_single[j];
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
    //initialize arrays
    // 1. node_attr (input)
    data_T node_attr[CONFIG_T::n_node][CONFIG_T::node_dim];
    nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::node_attr_config>(node_attr_1D, node_attr);

    // 2. edge_attr (input)
    data_T edge_attr[CONFIG_T::n_edge][CONFIG_T::edge_dim];
    nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::edge_attr_config>(edge_attr_1D, edge_attr);

    // 3. edge_index (input)
    index_T edge_index[CONFIG_T::n_edge][2];
    nnet::vec_to_mat<index_T, index_T, typename CONFIG_T::edge_index_config>(edge_index_1D, edge_index);
    if(CONFIG_T::io_stream){
      #pragma HLS STREAM variable=edge_index
    }

    // 4. num_edge_per_node (intermediate)
    index_T num_edge_per_node[CONFIG_T::n_node];
    for(int i=0; i<CONFIG_T::n_node; i++){
        num_edge_per_node[i] = 0;
    }

    // 5. edge_update (output)
    res_T edge_update[CONFIG_T::n_edge][CONFIG_T::out_dim];

    // 6. edge_update_aggr (output)
    res_T edge_update_aggr[CONFIG_T::n_node][CONFIG_T::out_dim];

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    edge_loop: for(int i = 0; i < CONFIG_T::n_edge; i++) { //for each edge
      #pragma HLS UNROLL

      // get sender, receiver indices
      index_T s;
      index_T r;
      if(CONFIG_T::flow==0){ //flow='source_to_target'
        s = edge_index[i][0];
        r = edge_index[i][1];
      }
      else{ //flow='target_to_source'
        s = edge_index[i][1];
        r = edge_index[i][0];
      }
      num_edge_per_node[r] += 1;

      // construct NN input: <receiver, sender, edge>
      data_T node_concat[2*CONFIG_T::node_dim];
      #pragma HLS ARRAY_PARTITION variable=node_concat complete dim=0
      nnet::concatenate1d<data_T, data_T, data_T, typename CONFIG_T::merge_config1>(node_attr[r], node_attr[s], node_concat);
      data_T phi_input[CONFIG_T::edge_dim + 2*CONFIG_T::node_dim];
      #pragma HLS ARRAY_PARTITION variable=phi_input complete dim=0
      nnet::concatenate1d<data_T, data_T, data_T, typename CONFIG_T::merge_config2>(node_concat, edge_attr[i], phi_input);

      // send it through NN
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
	      nnet::dense_mult_3lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update[i], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2);
        }
        else if(CONFIG_T::n_layers == 4){
	      nnet::dense_mult_4lyr<data_T, res_T, CONFIG_T>(phi_input, edge_update[i], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
        }
      }

      // aggregation step
      if(num_edge_per_node[r] <= 1){ //if this is the first edge sent to that index, there's nothing to aggregate with
        nnet::replace_single_edge<res_T, res_T, typename CONFIG_T::aggregation_config1>(edge_update[i], edge_update_aggr[r]);
      }
      else{
        if((CONFIG_T::aggr==0)||(CONFIG_T::aggr==1)){ //aggr="add" or "mean"
          nnet::aggregate_single_edge_add<res_T, res_T, typename CONFIG_T::aggregation_config1>(edge_update[i], edge_update_aggr[r]);
        }
        else if(CONFIG_T::aggr==2){ //aggr="max"
          nnet::aggregate_single_edge_max<res_T, res_T, typename CONFIG_T::aggregation_config1>(edge_update[i], edge_update_aggr[r]);
        }
      }
    }

    // taking care of edge_update_aggr
    res_T zeros[CONFIG_T::out_dim];
    for(int j=0; j<CONFIG_T::out_dim; j++){
          zeros[j]=0;
    }
    for(int i=0; i < CONFIG_T::n_node; i++){
      if(num_edge_per_node[i] < 1){ //disconnected nodes should have zeros in place of their edge_attr_aggr
        nnet::replace_single_edge<res_T, res_T, typename CONFIG_T::aggregation_config1>(zeros, edge_update_aggr[i]);
      }
      else if(CONFIG_T::aggr==1){ //if aggregation-method is "mean", we have to divide by the number of edges
        for (int j=0; j<CONFIG_T::out_dim; j++){
            edge_update_aggr[i][j] = edge_update_aggr[i][j]/num_edge_per_node[i];
        }
      }
    }

    //output arrays --> output vectors
    // 1. edge_update_1D
    nnet::mat_to_vec<res_T, res_T, typename CONFIG_T::edge_update_config>(edge_update, edge_update_1D);

    // 2. edge_update_aggr_1D
    nnet::mat_to_vec<res_T, res_T, typename CONFIG_T::edge_update_aggr_config>(edge_update_aggr, edge_update_aggr_1D);

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
    //initialize arrays
    //1. node_attr (input)
    data_T node_attr[CONFIG_T::n_node][CONFIG_T::node_dim];
    nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::node_attr_config>(node_attr_1D, node_attr);

    //2. edge_attr_aggr (input)
    data_T edge_attr_aggr[CONFIG_T::n_node][CONFIG_T::edge_dim];
    nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::edge_attr_aggr_config>(edge_attr_aggr_1D, edge_attr_aggr);

    // 3. node_update (output)
    res_T node_update[CONFIG_T::n_node][CONFIG_T::out_dim];

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    node_loop: for(int i = 0; i < CONFIG_T::n_node; i++){ //for each node
      #pragma HLS UNROLL

      // construct NN input: <node, edge_attr_aggr>
      data_T phi_input[CONFIG_T::edge_dim + CONFIG_T::node_dim];
      #pragma HLS ARRAY_PARTITION variable=phi_input complete dim=0
      nnet::concatenate1d<data_T, data_T, data_T, typename CONFIG_T::merge_config1>(node_attr[i], edge_attr_aggr[i], phi_input);

      // send it through NN
      if(CONFIG_T::activate_final){
	    data_T node_update_logits[CONFIG_T::node_dim];
	    #pragma HLS ARRAY_PARTITION variable=node_update_logits complete dim=0
	    if(CONFIG_T::n_layers == 1){
	      nnet::dense_mult_1lyr<data_T, data_T, CONFIG_T>(phi_input, node_update_logits, core_node_w0, core_node_b0);
	      nnet::relu<data_T, res_T, typename CONFIG_T::relu_config1>(node_update_logits, node_update[i]);
	    }
	    else if(CONFIG_T::n_layers == 2){
	      nnet::dense_mult_2lyr<data_T, data_T, CONFIG_T>(phi_input, node_update_logits, core_node_w0, core_node_b0, core_node_w1, core_node_b1);
	      nnet::relu<data_T, res_T, typename CONFIG_T::relu_config2>(node_update_logits, node_update[i]);
	    }
	    else if(CONFIG_T::n_layers == 3){
	      nnet::dense_mult_3lyr<data_T, data_T, CONFIG_T>(phi_input, node_update_logits, core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2);
	      nnet::relu<data_T, res_T, typename CONFIG_T::relu_config3>(node_update_logits, node_update[i]);
	    }
	    else if(CONFIG_T::n_layers == 4){
	      nnet::dense_mult_4lyr<data_T, data_T, CONFIG_T>(phi_input, node_update_logits, core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2, core_node_w3, core_node_b3);
	      nnet::relu<data_T, res_T, typename CONFIG_T::relu_config4>(node_update_logits, node_update[i]);
	    }
      }
      else{
        if(CONFIG_T::n_layers == 1){
	      nnet::dense_mult_1lyr<data_T, res_T, CONFIG_T>(phi_input, node_update[i], core_node_w0, core_node_b0);
        }
        else if(CONFIG_T::n_layers == 2){
	      nnet::dense_mult_2lyr<data_T, res_T, CONFIG_T>(phi_input, node_update[i], core_node_w0, core_node_b0, core_node_w1, core_node_b1);
        }
        else if(CONFIG_T::n_layers == 3){
	      nnet::dense_mult_3lyr<data_T, res_T, CONFIG_T>(phi_input, node_update[i], core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2);
        }
        else if(CONFIG_T::n_layers == 4){
	      nnet::dense_mult_4lyr<data_T, res_T, CONFIG_T>(phi_input, node_update[i], core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2, core_node_w3, core_node_b3);
        }
      }
    }

    // output array --> output vector
    nnet::mat_to_vec<res_T, res_T, typename CONFIG_T::node_update_config>(node_update, node_update_1D);

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

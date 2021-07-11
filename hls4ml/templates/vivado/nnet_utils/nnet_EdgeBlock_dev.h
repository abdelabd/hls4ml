
#include "nnet_common.h"
#include "nnet_merge.h"
#include "nnet_dense.h"
#include "nnet_dense_resource.h"
#include "nnet_activation.h"
#include "nnet_array.h"
#include "hls_stream.h"
#include "nnet_graph.h"
#include <string>
#include <sstream>
#include <math.h>

namespace nnet{

  template<class data_T, class index_T, class res_T, typename CONFIG_T>
  void EdgeBlock_dev(
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
    index_T num_edge_per_node[CONFIG_T::n_node];
    #pragma HLS ARRAY_PARTITION variable=num_edge_per_node complete dim=0
    for(int i=0; i<CONFIG_T::n_node; i++){
        num_edge_per_node[i] = 0;
    }

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    edge_loop: for(int i = 0; i < CONFIG_T::n_edge; i++) { //for each edge
      #pragma HLS UNROLL

      //get sender, receiver indices
      index_T s;
      index_T r;
      if(CONFIG_T::flow == source_to_target){
        s = edge_index_1D[2*i];
        r = edge_index_1D[2*i+1];
      }
      else{
        s = edge_index_1D[2*i+1];
        r = edge_index_1D[2*i];
      }
      num_edge_per_node[r] += 1;

      //get edge attributes
      data_T edge_i[CONFIG_T::edge_dim];
      for(int j=0; j<CONFIG_T::edge_dim; j++){
        #pragma HLS UNROLL
        edge_i[j] = edge_attr_1D[i*CONFIG_T::edge_dim+j];
      }

      //get sender, receiver attributes
      data_T node_s[CONFIG_T::node_dim];
      data_T node_r[CONFIG_T::node_dim];
      for(int j=0; j<CONFIG_T::node_dim; j++){
        #pragma HLS UNROLL
        node_s[j] = node_attr_1D[s*CONFIG_T::node_dim+j];
        node_r[j] = node_attr_1D[r*CONFIG_T::node_dim+j];
      }

      // construct NN input: <receiver, sender, edge>
      data_T node_concat[2*CONFIG_T::node_dim];
      #pragma HLS ARRAY_PARTITION variable=node_concat complete dim=0
      nnet::concatenate1d<data_T, data_T, data_T, typename CONFIG_T::merge_config1>(node_r, node_s, node_concat);
      data_T phi_input[CONFIG_T::edge_dim + 2*CONFIG_T::node_dim];
      #pragma HLS ARRAY_PARTITION variable=phi_input complete dim=0
      nnet::concatenate1d<data_T, data_T, data_T, typename CONFIG_T::merge_config2>(node_concat, edge_i, phi_input);

      // send it through NN
      res_T edge_update_i[CONFIG_T::out_dim];
      if(CONFIG_T::activate_final){
	    data_T edge_update_logits[CONFIG_T::out_dim];
        #pragma HLS ARRAY_PARTITION variable=edge_update_logits complete dim=0
        if(CONFIG_T::n_layers == 1){
	      nnet::dense_mult_1lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update_logits, core_edge_w0, core_edge_b0);
	      nnet::relu<data_T, res_T, typename CONFIG_T::relu_config1>(edge_update_logits, edge_update_i);
        }
        else if(CONFIG_T::n_layers == 2){
	      nnet::dense_mult_2lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update_logits, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1);
	      nnet::relu<data_T, res_T, typename CONFIG_T::relu_config2>(edge_update_logits, edge_update_i);
	    }
	    else if(CONFIG_T::n_layers == 3){
	      nnet::dense_mult_3lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update_logits, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2);
	      nnet::relu<data_T, res_T, typename CONFIG_T::relu_config3>(edge_update_logits, edge_update_i);
	    }
	    else if(CONFIG_T::n_layers == 4){
	      nnet::dense_mult_4lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update_logits, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
	      nnet::relu<data_T, res_T, typename CONFIG_T::relu_config4>(edge_update_logits, edge_update_i);
	    }
      }
      else{
        if(CONFIG_T::n_layers == 1){
	      nnet::dense_mult_1lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update_i, core_edge_w0, core_edge_b0);
          }
        else if(CONFIG_T::n_layers == 2){
	      nnet::dense_mult_2lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update_i, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1);
        }
        else if(CONFIG_T::n_layers == 3){
	      nnet::dense_mult_3lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update_i, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2);
        }
        else if(CONFIG_T::n_layers == 4){
	      nnet::dense_mult_4lyr<data_T, res_T, CONFIG_T>(phi_input, edge_update_i, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
        }
      }

      //send edge_update_i-->edge_update_1D, edge_update_aggr_1D
      for(int j=0; j<CONFIG_T::out_dim; j++){
        #pragma HLS UNROLL

        //edge_update
        edge_update_1D[i*CONFIG_T::out_dim+j] = edge_update_i[j];

        //edge_update_aggr
        if(num_edge_per_node[r] <= 1){
          edge_update_aggr_1D[r*CONFIG_T::out_dim+j] = edge_update_i[j];
        }
        else{
          if(CONFIG_T::aggr == aggr_sum || CONFIG_T::aggr == aggr_mean){
            edge_update_aggr_1D[r*CONFIG_T::out_dim+j] += edge_update_i[j];
          }
          else{ //aggr=max
            edge_update_aggr_1D[r*CONFIG_T::out_dim+j] =  edge_update_i[j] > edge_update_aggr_1D[r*CONFIG_T::out_dim+j] ? edge_update_i[j] : edge_update_aggr_1D[r*CONFIG_T::out_dim+j];
          }
        }
      }
    }

    for(int i=0; i<CONFIG_T::n_node; i++){
      if(num_edge_per_node[i] < 1){
        for(int j=0; j<CONFIG_T::out_dim; j++){
          #pragma HLS UNROLL
          edge_update_aggr_1D[i*CONFIG_T::out_dim+j] = 0;
        }
      }
      else{
        if(CONFIG_T::aggr==1){ //if aggregation-method is "mean", we have to divide by the number of edges
          for (int j=0; j<CONFIG_T::out_dim; j++){
            #pragma HLS UNROLL
            res_T edge_mean_j;
            nnet::edge_divide<res_T, index_T, res_T, CONFIG_T>(edge_update_aggr_1D[i*CONFIG_T::out_dim+j], num_edge_per_node[i], edge_mean_j);
            edge_update_aggr_1D[i*CONFIG_T::out_dim+j] = edge_mean_j;
          }
        }
      }
    }

  }

}
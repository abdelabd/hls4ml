#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "firmware/myproject.h"
#include "firmware/nnet_utils/nnet_helpers.h"

#define CHECKPOINT 5000

namespace nnet {
    bool trace_enabled = true;
    std::map<std::string, void *> *trace_outputs = NULL;
    size_t trace_type_size = sizeof(double);
}

int main(int argc, char **argv)
{
  //load input data from text file
  std::ifstream fin1("tb_data/tb_input_edge_features.dat");
  std::ifstream fin2("tb_data/tb_input_node_features.dat");
  std::ifstream fin3("tb_data/tb_input_edge_index.dat");
  //load predictions from text file
  std::ifstream fpr("tb_data/tb_output_predictions.dat");

#ifdef RTL_SIM
  std::string RESULTS_LOG = "tb_data/rtl_cosim_results.log";
#else
  std::string RESULTS_LOG = "tb_data/csim_results.log";
#endif
  std::ofstream fout(RESULTS_LOG);

  std::string iline1, iline2, iline3;
  std::string pline;
  int e = 0;

  if (fin1.is_open() && fin2.is_open() && fin3.is_open() && fpr.is_open()) {
    while ( std::getline(fin1,iline1) && std::getline(fin2,iline2) && std::getline(fin3,iline3) && std::getline (fpr,pline) ) {
      if (e % CHECKPOINT == 0) std::cout << "Processing input " << e << std::endl;
      char* cstr=const_cast<char*>(iline1.c_str());
      char* current;
      std::vector<float> in1;
      current=strtok(cstr," ");
      while(current!=NULL) {
        in1.push_back(atof(current));
        current=strtok(NULL," ");
      }
      cstr=const_cast<char*>(iline2.c_str());
      std::vector<float> in2;
      current=strtok(cstr," ");
      while(current!=NULL) {
        in2.push_back(atof(current));
        current=strtok(NULL," ");
      }
      cstr=const_cast<char*>(iline3.c_str());
      std::vector<float> in3;
      current=strtok(cstr," ");
      while(current!=NULL) {
        in3.push_back(atof(current));
        current=strtok(NULL," ");
      }
      cstr=const_cast<char*>(pline.c_str());
      std::vector<float> pr;
      current=strtok(cstr," ");
      while(current!=NULL) {
        pr.push_back(atof(current));
        current=strtok(NULL," ");
      }

      //hls-fpga-machine-learning insert data

      //hls-fpga-machine-learning insert top-level-function

      if (e % CHECKPOINT == 0) {
        std::cout << "Predictions" << std::endl;
        //hls-fpga-machine-learning insert predictions
        std::cout << "Quantized predictions" << std::endl;
        //hls-fpga-machine-learning insert quantized
      }
      e++;

      //hls-fpga-machine-learning insert tb-output

    }
    fin1.close();
    fin2.close();
    fin3.close();
    fpr.close();
  } else {
    std::cout << "INFO: Unable to open input/predictions file, using default input." << std::endl;
    //hls-fpga-machine-learning insert zero

    //hls-fpga-machine-learning insert top-level-function

    //hls-fpga-machine-learning insert output

    //hls-fpga-machine-learning insert tb-output
  }

  fout.close();
  std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

  return 0;
}

//Numpy array shape [1, 1]
//Min 0.0
//Max 0.0
//Number of zeros 1

#ifndef CORE_NODE_W2_H_
#define CORE_NODE_W2_H_

#ifndef __SYNTHESIS__
ap_fixed<16,6> core_node_w2[1];
#else
ap_fixed<16,6> core_node_w2[1] = {0.0};
#endif

#endif
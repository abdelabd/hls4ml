//Numpy array shape [8, 8]
//Min -0.711631146312
//Max 0.946795503191
//Number of zeros 0

#ifndef DECODER_EDGE_W2_H_
#define DECODER_EDGE_W2_H_

#ifndef __SYNTHESIS__
ap_fixed<16,6> decoder_edge_w2[64];
#else
ap_fixed<16,6> decoder_edge_w2[64] = {-0.538910, 0.402150, 0.197081, -0.329331, -0.241922, -0.024030, 0.686408, -0.103200, -0.065314, -0.016667, 0.335049, -0.482623, -0.048546, 0.267885, 0.050295, -0.102084, 0.908440, -0.711631, 0.466638, 0.582447, -0.068337, -0.234975, 0.301474, -0.590367, -0.016153, 0.500033, 0.017994, 0.031812, 0.642546, 0.339126, 0.078172, 0.402354, -0.077995, 0.032996, 0.404073, 0.060761, 0.308730, 0.822308, -0.015548, 0.471865, -0.278250, 0.671211, 0.373123, 0.291633, 0.946796, 0.443234, -0.591846, 0.709861, 0.115091, 0.491747, -0.361419, -0.309020, 0.320312, 0.274361, 0.405378, 0.516258, -0.361182, 0.498058, -0.178861, 0.362186, 0.284953, 0.519844, 0.297514, 0.551636};
#endif

#endif

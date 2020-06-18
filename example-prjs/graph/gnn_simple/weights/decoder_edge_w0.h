//Numpy array shape [8, 8]
//Min -0.830038360077
//Max 1.099603200655
//Number of zeros 0

#ifndef DECODER_EDGE_W0_H_
#define DECODER_EDGE_W0_H_

#ifndef __SYNTHESIS__
ap_fixed<16,6> decoder_edge_w0[64];
#else
ap_fixed<16,6> decoder_edge_w0[64] = {0.016552, 0.337231, -0.024060, 0.726877, 0.007309, -0.516642, 0.658639, -0.625100, 0.422458, 0.288824, 0.118533, 0.500818, 0.789117, 0.142093, 0.513733, -0.343859, 0.046643, 0.284580, -0.558461, 0.690935, 0.695126, -0.830038, 0.281458, -0.380099, 0.157212, 0.405758, -0.206790, 0.576522, 0.537497, 1.099603, -0.096785, -0.320486, 0.541490, -0.184953, 0.326005, 0.046956, -0.053611, 0.766268, -0.232064, -0.107203, -0.219946, 0.157099, -0.061843, 0.528132, -0.114937, -0.227893, 0.598729, -0.092851, -0.146185, -0.204822, 0.644249, 0.632283, 0.834409, 0.567219, 0.406252, 0.123093, -0.126219, 0.535304, -0.054847, 0.458463, 0.107895, -0.583256, 0.005013, 0.326249};
#endif

#endif

set arg_0 "-I . -DN_INPUT=N_FILT_28 -DN_OUTPUT=N_FILT_31"
set arg_1 "-DCONFIG=config31"
set arg_2 "-DINPUT_T=layer30_t -DLAYER_T=layer33_t"
set arg_3 "-DN_WEIGHTS=18432 -DWEIGHTS=w31  -DBIASES=b31"
set args "$arg_0 $arg_1 $arg_2 $arg_3"
set layer_type conv_2d_large_cl_port


source ../common/build.tcl


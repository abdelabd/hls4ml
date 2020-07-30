//Numpy array shape [32, 32]
//Min -0.352011153399
//Max 0.351719939960
//Number of zeros 0

#ifndef DECODER_EDGE_W1_H_
#define DECODER_EDGE_W1_H_

#ifndef __SYNTHESIS__
ap_fixed<16,6> decoder_edge_w1[1024];
#else
ap_fixed<16,6> decoder_edge_w1[1024] = {-0.001329, 0.263558, -0.004892, -0.126987, 0.142169, 0.315527, -0.016523, -0.036247, 0.057381, 0.076137, 0.287337, 0.235146, 0.254376, 0.129817, -0.285101, -0.023724, -0.013031, -0.035415, -0.125574, -0.287354, -0.104810, -0.080563, 0.244302, -0.187430, 0.328716, -0.240179, -0.058978, -0.075681, -0.122049, 0.053733, 0.269074, 0.203092, 0.165042, -0.243784, 0.089169, -0.142902, -0.140598, 0.047141, 0.073078, -0.176237, -0.062528, 0.124799, 0.072017, -0.040814, 0.275964, 0.182877, 0.243761, -0.060499, 0.150170, -0.017098, -0.129301, 0.040587, -0.050190, 0.074142, -0.256861, -0.139331, -0.042159, 0.026198, 0.047526, -0.231751, 0.074749, -0.079714, -0.087680, -0.170393, -0.271161, 0.109806, 0.289478, 0.067087, 0.000908, 0.090716, -0.019555, -0.012386, -0.342491, -0.104819, -0.017676, -0.106669, -0.131362, 0.146481, 0.183380, 0.027609, -0.210459, -0.056042, -0.059686, 0.074700, 0.108164, 0.005532, -0.015296, -0.130127, 0.129532, 0.064164, -0.218801, 0.173771, 0.073236, 0.111937, -0.039776, -0.075825, 0.120027, -0.323018, -0.088898, 0.175938, -0.159871, -0.214174, 0.159354, 0.185987, 0.167367, -0.166591, -0.046898, 0.305046, 0.200575, -0.082997, -0.224344, -0.249159, -0.032326, -0.157448, -0.071865, -0.118735, 0.195250, 0.156335, -0.199476, 0.044535, -0.230644, 0.215865, 0.002101, -0.032107, 0.097654, -0.107825, 0.222565, -0.060340, 0.043043, -0.000874, 0.047136, 0.004062, 0.016522, -0.070258, -0.275256, -0.160294, 0.147053, -0.090656, 0.001151, 0.085458, -0.029855, 0.122920, 0.111231, 0.091503, -0.244459, 0.323947, -0.181652, -0.245522, -0.093621, 0.003799, -0.046558, 0.246397, 0.128016, 0.233122, -0.090657, -0.077688, 0.041402, 0.071705, -0.309850, -0.066416, -0.030878, -0.190846, 0.095172, -0.053734, -0.062673, -0.131659, 0.328205, -0.199523, -0.037824, 0.127781, 0.199415, -0.047997, 0.069477, -0.213514, -0.192411, 0.004651, -0.142322, 0.021211, 0.221123, -0.239511, -0.100945, 0.112949, -0.146414, 0.040417, -0.010886, 0.046441, 0.010107, 0.293295, -0.030938, 0.126442, 0.066986, -0.040493, -0.154590, 0.094829, -0.241845, 0.023454, -0.186701, 0.033774, -0.080401, -0.163606, -0.007162, 0.009669, 0.216598, -0.020475, -0.151770, -0.207941, 0.118576, -0.021630, -0.019788, 0.064976, 0.292710, -0.231208, -0.008164, 0.187456, 0.052831, 0.081064, -0.020489, -0.065178, -0.094533, -0.037960, -0.308901, -0.196559, 0.006291, 0.037316, 0.103501, -0.039285, -0.163106, 0.131404, -0.176679, 0.065496, -0.177951, -0.009777, 0.300567, 0.057917, -0.196499, 0.293353, -0.218208, -0.015933, 0.066453, -0.263664, 0.095842, -0.188841, -0.074239, -0.139538, 0.069476, -0.050081, 0.159349, -0.235809, 0.063792, -0.230129, 0.042998, -0.056938, 0.166624, 0.295966, -0.098159, 0.216495, 0.271444, -0.197520, -0.064381, 0.111110, -0.266250, 0.044380, -0.008128, 0.067683, -0.157271, -0.002574, 0.197739, -0.102266, -0.073046, 0.066276, 0.078930, 0.098480, -0.166250, 0.000266, -0.083887, -0.194620, 0.106633, -0.159287, 0.232186, -0.207067, -0.119532, 0.279882, 0.145067, 0.067407, 0.212649, 0.011958, 0.110305, -0.160165, -0.188472, 0.302142, 0.068653, 0.004753, 0.144961, -0.068733, 0.035163, -0.172234, 0.051272, 0.001832, 0.070771, 0.034394, 0.182465, 0.019623, 0.052863, 0.164819, -0.099371, 0.107676, 0.099979, 0.169640, 0.138035, -0.141431, -0.268236, 0.130485, 0.044789, -0.182169, 0.000365, 0.086265, 0.011725, -0.024611, 0.074739, 0.047838, -0.009426, -0.189145, 0.112012, 0.051275, -0.062375, -0.062927, 0.279831, -0.327257, -0.031775, 0.042070, -0.166381, -0.071959, -0.131233, -0.063623, -0.039425, 0.048037, -0.044408, 0.060843, -0.167141, -0.050090, -0.154204, -0.146942, 0.276213, -0.251963, 0.151635, -0.027475, 0.106257, -0.058545, 0.164377, 0.032337, -0.002033, 0.029180, -0.079706, -0.029230, 0.010288, 0.119710, 0.012195, 0.029217, -0.187379, -0.124052, 0.106744, 0.031483, 0.033827, -0.022829, 0.033057, -0.003032, -0.262707, 0.262925, -0.180571, 0.108820, 0.273302, -0.011471, 0.307590, -0.059145, 0.213878, 0.012111, -0.005170, 0.064308, 0.005878, 0.076620, 0.101725, -0.015558, -0.105432, -0.021973, 0.049253, -0.316432, 0.169941, -0.114883, 0.063545, 0.096512, -0.052583, -0.065877, -0.077478, 0.307295, -0.005532, 0.246783, -0.091666, -0.118684, 0.138180, -0.051483, 0.171223, -0.140743, -0.010492, 0.124783, 0.033006, -0.080036, 0.147147, -0.073049, 0.021134, 0.052850, -0.067492, -0.067606, 0.168373, 0.075671, 0.043346, 0.078584, 0.142328, -0.001830, 0.195019, 0.118923, 0.022182, -0.069201, 0.225022, 0.135652, 0.198428, 0.154535, 0.002647, -0.148071, 0.127567, -0.054063, -0.192763, 0.260144, -0.093433, 0.228773, -0.221921, 0.172291, -0.063451, -0.157709, -0.177564, 0.012912, -0.082351, 0.106374, -0.185508, 0.135535, 0.309020, -0.197030, 0.288839, -0.028102, 0.130862, -0.072564, 0.078790, -0.007034, 0.284199, 0.006671, 0.107121, -0.182332, 0.093129, -0.253777, 0.053118, 0.351037, -0.185693, 0.042564, -0.319049, 0.049715, 0.116316, -0.137781, -0.292952, 0.251016, 0.010701, -0.227934, -0.183929, -0.118573, 0.023751, -0.173905, 0.183366, -0.287111, 0.054555, 0.059900, -0.337277, 0.152507, -0.215051, -0.082330, 0.337759, 0.196111, 0.242224, -0.045687, 0.009456, 0.108272, 0.060999, 0.174668, 0.221457, -0.125885, -0.003077, -0.223523, -0.059624, -0.024008, -0.189387, -0.007442, -0.113347, 0.018612, 0.109050, -0.253138, 0.097836, -0.109629, -0.059813, 0.029322, 0.107306, -0.115939, 0.083499, 0.083190, 0.082350, 0.091997, -0.310765, -0.135998, -0.056409, 0.157636, -0.009823, 0.065868, -0.056064, -0.116989, 0.032176, -0.166849, -0.323316, 0.016393, -0.126190, -0.029234, -0.107796, 0.219413, 0.186377, -0.014176, 0.149327, -0.056940, -0.008255, 0.029682, -0.012189, -0.014997, -0.103929, -0.139892, -0.012512, 0.202297, 0.083607, -0.204653, -0.077031, -0.088716, -0.145893, 0.144148, 0.120063, 0.047246, -0.123869, 0.172338, 0.027206, -0.208186, 0.209054, 0.029903, 0.027435, 0.050499, -0.244926, -0.075464, -0.187024, -0.284673, 0.003318, 0.270971, -0.027849, 0.051564, 0.120150, 0.133433, -0.028600, -0.193824, -0.154087, 0.171258, -0.005073, 0.111800, -0.116540, -0.024953, -0.152016, 0.085138, -0.023140, -0.123380, 0.061695, -0.004039, 0.047420, 0.137778, 0.256411, 0.111915, -0.018739, -0.060125, -0.187858, -0.032508, -0.149461, -0.053307, 0.150978, 0.108659, 0.164562, 0.055400, 0.108371, -0.252552, -0.057981, -0.018014, 0.025979, 0.247794, 0.163673, -0.114026, -0.273277, 0.130706, 0.197279, 0.067871, -0.080568, 0.015626, 0.042056, 0.203394, 0.078785, -0.078545, -0.172990, -0.329164, 0.046267, -0.136666, -0.127426, 0.075082, 0.084211, -0.097083, 0.108466, -0.306559, 0.109779, -0.233956, 0.043164, -0.009650, -0.029428, 0.012852, -0.191423, -0.323434, -0.277413, 0.060864, 0.207112, -0.067650, 0.217861, -0.267378, 0.043183, -0.227036, -0.028773, -0.127940, 0.301739, -0.100799, -0.055979, -0.120828, 0.067304, -0.172593, -0.352011, -0.041487, -0.199687, -0.140287, -0.178617, 0.156912, 0.170970, 0.058380, 0.263207, -0.127221, -0.209371, 0.279149, -0.196300, -0.089694, 0.200275, -0.072308, -0.350612, -0.141902, -0.343131, 0.151398, 0.141188, -0.049988, 0.262332, 0.231651, -0.062858, 0.318946, 0.286003, 0.131342, 0.095896, -0.016428, 0.176900, 0.339812, -0.004494, -0.000799, 0.263025, 0.129121, -0.029502, -0.023260, 0.132474, -0.182653, -0.010311, 0.216507, 0.001987, -0.272204, 0.085216, -0.012078, 0.165237, -0.054095, -0.188447, -0.214106, 0.232936, -0.054547, 0.118799, -0.242779, 0.111853, 0.241982, -0.085182, -0.160211, -0.005855, -0.016340, 0.185284, -0.150629, 0.146694, -0.121659, -0.217823, -0.164207, -0.144588, -0.059501, -0.087370, -0.102658, -0.011077, 0.244389, 0.229056, 0.107511, -0.291671, -0.047878, -0.261901, -0.108792, 0.236079, -0.051772, 0.119635, -0.263329, -0.075082, -0.220849, -0.076681, 0.156282, 0.223470, 0.137789, 0.163847, 0.169592, -0.072242, -0.222637, -0.239934, -0.261489, 0.047479, -0.135479, -0.185086, -0.039355, 0.132433, 0.293439, 0.000286, 0.036359, -0.126301, -0.217851, 0.006267, -0.249325, -0.154135, 0.041256, -0.167107, 0.151146, 0.106611, 0.265868, 0.316191, 0.079339, -0.238261, 0.070719, -0.070252, -0.062573, 0.144996, 0.350997, 0.298467, 0.209833, 0.190150, 0.062823, -0.101843, 0.109235, 0.072374, 0.000950, -0.145489, -0.098074, -0.137340, -0.020806, 0.030113, -0.292473, -0.168129, -0.106789, -0.166924, -0.243492, -0.046355, 0.158033, -0.043714, -0.227006, -0.061678, -0.148005, 0.105342, 0.138028, 0.126215, 0.007070, -0.195607, 0.086602, 0.108357, -0.067199, -0.250746, 0.044846, 0.105174, 0.033215, -0.109942, -0.031153, 0.223867, -0.098933, -0.262315, -0.159660, -0.016637, -0.016134, -0.192650, -0.213810, 0.118529, 0.003177, 0.018508, 0.063457, -0.238550, -0.111297, -0.036009, -0.079544, 0.140897, -0.184888, 0.091441, 0.057106, 0.186007, -0.049205, 0.237652, 0.088656, 0.121234, -0.123390, 0.167657, -0.094165, 0.103785, -0.145155, -0.132611, -0.031701, 0.049766, -0.109151, 0.258872, -0.139017, -0.071918, 0.109217, 0.071462, 0.091434, -0.293581, 0.200942, 0.002956, -0.137870, 0.053312, 0.038025, -0.071671, 0.058018, 0.015495, -0.089152, 0.100533, -0.008062, -0.072582, -0.147175, -0.317045, -0.231077, 0.181256, 0.238843, 0.179574, -0.007426, 0.067490, -0.036363, -0.055662, 0.253018, -0.279037, 0.174247, -0.015517, 0.264557, -0.091223, -0.087156, 0.140277, -0.048281, 0.057604, 0.213517, 0.053780, -0.317969, 0.050676, 0.197268, -0.238448, -0.085657, 0.085122, 0.011909, 0.058223, -0.028390, 0.250560, 0.129890, -0.031804, 0.346058, -0.029211, 0.010744, -0.099782, -0.013767, 0.085093, -0.059923, 0.057576, 0.019617, -0.002885, -0.306808, 0.080484, 0.010690, -0.067097, 0.121306, -0.141025, 0.032680, -0.021598, -0.029550, 0.060948, -0.278917, 0.035633, -0.048274, 0.148171, -0.139892, 0.173609, 0.146478, -0.053528, -0.076351, -0.128673, -0.183129, -0.165285, 0.050248, 0.012542, -0.051614, -0.062388, -0.108936, -0.104405, -0.129646, -0.190660, -0.092729, 0.129646, -0.154286, 0.058738, -0.021996, -0.025606, 0.047319, -0.215731, -0.210305, 0.273482, -0.173838, -0.165699, -0.098589, -0.004317, -0.012231, -0.071518, 0.143233, -0.184532, 0.345282, -0.051692, 0.154263, -0.074156, -0.102909, -0.199374, -0.087408, -0.018555, -0.080248, 0.225865, -0.087363, -0.224301, 0.115795, 0.026935, -0.177581, -0.148996, -0.200515, -0.059100, 0.020288, -0.014313, 0.279393, 0.141646, -0.170142, -0.336614, 0.351720, -0.138140, -0.011626, -0.331736, 0.274266, -0.326099, -0.022392, -0.240980, -0.014452, 0.124220, 0.179584, 0.270172, -0.028649, 0.296998, -0.182710, 0.070815, -0.120904, 0.024176, -0.084284, 0.012542, -0.093622, 0.232873, -0.300560, 0.058557, -0.257257, -0.016719, 0.129079, 0.027993, 0.152984, 0.047895, 0.266775, -0.270718, -0.143818, -0.020121, -0.065836, -0.246706, 0.041143, -0.256886, -0.212544, -0.009761, -0.085660, -0.189431, 0.051146, 0.054328, 0.094531, 0.296448, 0.099567, -0.204136, -0.045128, -0.100931, 0.277528};
#endif

#endif

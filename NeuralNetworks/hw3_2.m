w_i_hb = [ -4.8 5.0 7.1 ; -4.8 5.0 -2.1 ];
w_h_ob = [ 7.4 -5.2 -0.4 ; -3.1 -5.3 7.7 ];

o_h1 = logsig(w_i_hb*[0;0;1]);
o_o1 = logsig(w_h_ob*[o_h1;1])

o_h1 = logsig(w_i_hb*[1;0;1]);
o_o2 = logsig(w_h_ob*[o_h1;1])

o_h1 = logsig(w_i_hb*[0;1;1]);
o_o3 = logsig(w_h_ob*[o_h1;1])

o_h1 = logsig(w_i_hb*[1;1;1]);
o_o4 = logsig(w_h_ob*[o_h1;1])
w_i_hb = [ 1.4 0.4 0.0 ; -2.0 0.8 -0.6 ];
w_h_ob = [ 2.1 -1.0 0.4 ; 1.0 1.1 -0.3 ];
p=[-1;1];
d = [0.3;0.3];
alfa=1.5;

o_h = logsig(2*w_i_hb*[p;1])
o_o = logsig(2*w_h_ob*[o_h;1])

delta_o = (d-o_o)'*2*o_o*(ones(size(o_o))-o_o)



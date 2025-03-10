;-------------------------------------------------------------------------------
; Quick pro to take R, Theta, Phi and give xyz coords
;-------------------------------------------------------------------------------

function rtp2xyz, Rtp_D

r_=0
t_=1
p_=2

x_=0
y_=1
z_=2

Xyz_D = dblarr(3)

Xyz_D[x_] = Rtp_D[r_]*cos(Rtp_D[p_])*sin(Rtp_D[t_])
Xyz_D[y_] = Rtp_D[r_]*sin(Rtp_D[p_])*sin(Rtp_D[t_])
Xyz_D[z_] = Rtp_D[r_]*cos(Rtp_D[t_])

return, Xyz_D

end

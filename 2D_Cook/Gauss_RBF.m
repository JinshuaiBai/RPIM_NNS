function [u, u_x, u_y]=Gauss_RBF(x,c,h)

r=sqrt((x(1)-c(1))^2+(x(2)-c(2))^2);
b=9/4*(1/h)^2;
u=exp(-b*r^2);
tmp=-2*b*exp(-b*r^2);
u_x=tmp*(x(1)-c(1));
u_y=tmp*(x(2)-c(2));
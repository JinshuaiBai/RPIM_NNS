clc,clear,close all
%% Load sample points and data
% Data in the "Coord.mat" include:
% [x]       : Coordinates of in-domain sample points;
% [x_l]     : Coordinates of sample points on the left boundary;
% [x_r]     : Coordinates of sample points on the right boundary;
% [c]       : Center's coordinates of RBFs;
% [dV]      : Weights of in-domain sample points for quadature;
% [dS]      : Weight of the sample points on the right boundary for
%             quadrature;
% [P]       : Traction vector on the right boundary;
load Coord.mat

%% Initialise parameters
% The Young's modulus and the Poisson's ratio
E=5e2;nu=0.35;

% The Lame constants
la=E*nu/(1+nu)/(1-2*nu);
mu=E/(1+nu)/2;

% Support length
h=3;

% RBF type: 1 == Gaussian type RBF (Eq. 4 in the manuscript)
%           2 == Fifth order piecewise RBF (Eq. 5 in the manuscript)
RBF_type=2;

% Penalty value for imposing displacement boudnary condition
alpha=1e6;

% Maximum number of training iteration for convergence
max_iter=50000;

% Number of iteration for visualisation
vis_step=100;

% Initialise a matrix to store loss history
L_hist=[];

%% Initialised a
% a_u is for displacement u
a_u=zeros(length(c),1);

% a_v is for displacement v
a_v=zeros(length(c),1);

%% Initialise Adam optimiser
% Learning rate
lr=1e-4;

% Other parameters in Adam
beta1=0.9;
beta2=0.999;
e=1e-8;
m1=zeros(2*length(a_u),1);
m2=zeros(2*length(a_u),1);

%% Nearest search
% In-domain sample points
idx_c=[];
idx_x=[];
k=0;
for j=1:length(x)
    for i=1:length(c)
        if RBF_type==1
            k=k+1;
            [tmp(k), tmp_x(k), tmp_y(k)]=Gauss_RBF(x(j,:),c(i,:),h);
            idx_c(k)=i;
            idx_x(k)=j;
        elseif RBF_type==2
            rr=(x(j,1)-c(i,1))^2+(x(j,2)-c(i,2))^2;
            if rr<=4*h^2
                k=k+1;
                [tmp(k), tmp_x(k), tmp_y(k)]=FifthPiecewise_RBF(x(j,:),c(i,:),h);
                idx_c(k)=i;
                idx_x(k)=j;
            end
        else
            error('Please select correct RBF! You can choose:1. Gaussian type RBF (k_type = 1)2. Fifth order piecewise RBF (k_type = 2)\n');
        end
    end
end

% Sample points on the left boundary
idx_c_l=[];
idx_x_l=[];
k=0;
for i=1:length(c)
    for j=1:length(x_l)
        if RBF_type==1
            k=k+1;
            [tmp_l(k), ~, ~]=Gauss_RBF(x_l(j,:),c(i,:),h);
            idx_c_l(k)=i;
            idx_x_l(k)=j;
        elseif RBF_type==2
            rr=(x_l(j,1)-c(i,1))^2+(x_l(j,2)-c(i,2))^2;
            if rr<=4*h^2
                k=k+1;
                [tmp_l(k), ~, ~]=FifthPiecewise_RBF(x_l(j,:),c(i,:),h);
                idx_c_l(k)=i;
                idx_x_l(k)=j;
            end
        end
    end
end

% Sample points on the right boundary
idx_c_r=[];
idx_x_r=[];
k=0;
for i=1:length(c)
    for j=1:length(x_r)
        if RBF_type==1
            k=k+1;
            [tmp_r(k), ~, ~]=Gauss_RBF(x_r(j,:),c(i,:),h);
            idx_c_r(k)=i;
            idx_x_r(k)=j;
        elseif RBF_type==2
            rr=(x_r(j,1)-c(i,1))^2+(x_r(j,2)-c(i,2))^2;
            if rr<=4*h^2
                k=k+1;
                [tmp_r(k), ~, ~]=FifthPiecewise_RBF(x_r(j,:),c(i,:),h);
                idx_c_r(k)=i;
                idx_x_r(k)=j;
            end
        end
    end
end

%% Start training iteration
tic;
for t=1:max_iter
    %% In-domain points
    % Initialise displacement and deformation tensor
    u=zeros(length(x),1);
    v=zeros(length(x),1);
    ux=zeros(length(x),1);
    uy=zeros(length(x),1);
    vx=zeros(length(x),1);
    vy=zeros(length(x),1);

    % Calculate displacement and deformation tensor
    for loop=1:length(idx_x)
        i=idx_c(loop);
        j=idx_x(loop);

        u(j)=u(j)+a_u(i)*tmp(loop);
        v(j)=v(j)+a_v(i)*tmp(loop);
        ux(j)=ux(j)+a_u(i)*tmp_x(loop);
        uy(j)=uy(j)+a_u(i)*tmp_y(loop);
        vx(j)=vx(j)+a_v(i)*tmp_x(loop);
        vy(j)=vy(j)+a_v(i)*tmp_y(loop);
    end
    
    % Calculate the determinant of the deformation tensor
    J=(1+ux).*(1+vy)-uy.*vx;

    % Calculate the 1st invariant of the deformation tensor
    I=(1+ux).^2+(1+vy).^2+uy.^2+vx.^2;

    % Calculate gradients of potential energy towards a_u and a_v
    Grad_u_in=zeros(length(c),1);
    Grad_v_in=zeros(length(c),1);
    for loop=1:length(idx_c)
        i=idx_c(loop);
        j=idx_x(loop);

        dJdau=(1+vy(j))*tmp_x(loop)-vx(j)*tmp_y(loop);
        dJdav=(1+ux(j))*tmp_y(loop)-uy(j)*tmp_x(loop);
        dIdau=2*(1+ux(j))*tmp_x(loop)+2*uy(j)*tmp_y(loop);
        dIdav=2*(1+vy(j))*tmp_y(loop)+2*vx(j)*tmp_x(loop);

        % Grad_u_in(i)=Grad_u_in(i)+(la*log(J(j))/J(j)-mu/J(j))*dJdau*dV(j);
        % Grad_u_in(i)=Grad_u_in(i)+(mu/2)*dIdau*dV(j);
        % 
        % Grad_v_in(i)=Grad_v_in(i)+(la*log(J(j))/J(j)-mu/J(j))*dJdav*dV(j);
        % Grad_v_in(i)=Grad_v_in(i)+(mu/2)*dIdav*dV(j);

        Grad_u_in(i)=Grad_u_in(i)+(0.5*la*J(j)-(la/2+mu)/J(j))*dJdau*dV(j);
        Grad_u_in(i)=Grad_u_in(i)+(mu/2)*dIdau*dV(j);

        Grad_v_in(i)=Grad_v_in(i)+(0.5*la*J(j)-(la/2+mu)/J(j))*dJdav*dV(j);
        Grad_v_in(i)=Grad_v_in(i)+(mu/2)*dIdav*dV(j);
    end

    %% Sample points on the left boundary
    % Initialise displacement on the left boundary
    u_l=zeros(length(x_l),1);
    v_l=zeros(length(x_l),1);

    % Calculate displacement on the left boundary
    for loop=1:length(idx_x_l)
        i=idx_c_l(loop);
        j=idx_x_l(loop);

        u_l(j)=u_l(j)+a_u(i)*tmp_l(loop);
        v_l(j)=v_l(j)+a_v(i)*tmp_l(loop);
    end

    % Residual of displacement on the left boundary
    l_l_u=u_l-0;
    l_l_v=v_l-0;

    % Calculate gradients of residual on the left boundary
    Grad_u_l=zeros(length(c),1);
    Grad_v_l=zeros(length(c),1);
    for loop=1:length(idx_c_l)
        i=idx_c_l(loop);
        j=idx_x_l(loop);

        Grad_u_l(i)=Grad_u_l(i)+2*l_l_u(j)*tmp_l(loop);
        Grad_v_l(i)=Grad_v_l(i)+2*l_l_v(j)*tmp_l(loop);
    end

    %% Sample points on the right boundary
    % Initialise displacement on the right boundary
    v_r=zeros(length(x_r),1);
    for loop=1:length(idx_x_r)
        i=idx_c_r(loop);
        j=idx_x_r(loop);

        v_r(j)=v_r(j)+a_v(i)*tmp_r(loop);
    end

    % Calculate gradients of the traction boundary condition
    Grad_v_r=zeros(length(c),1);
    for loop=1:length(idx_c_r)
        i=idx_c_r(loop);
        j=idx_x_r(loop);

        Grad_v_r(i)=Grad_v_r(i)+tmp_r(loop)*dS*P(j);
    end

    %% Calculate loss and gradient
    % Calculate strain energy (internal energy)
    % E_in=sum((0.5*la*log(J).^2-mu*log(J)+0.5*mu*(I-2)).*dV);
    E_in=sum((0.25*la*(J.^2-1)-(la/2+mu)*log(J)+0.5*mu*(I-2)).*dV);

    % Calculate potential energy of external force
    E_ex=sum(v_r.*P*dS);

    % Calculate residual of displacement boundary
    R_disp=mean(l_l_u.^2)+mean(l_l_v.^2);

    % Calculate overall loss functional of the mechanics system
    L=E_in-E_ex+R_disp;

    % Summarise all graidents
    L_grad_u=Grad_u_in+Grad_u_l*alpha;
    L_grad_v=Grad_v_in-Grad_v_r+Grad_v_l*alpha;

    % Update loss history:  1st colume == overall loss
    %                       2nd colume == strain energy
    %                       3rd colume == potential energy of external
    %                       force
    %                       4th colume == residual of displacement boundary
    %                       conditions
    L_hist=[L_hist,;[L E_in E_ex R_disp]];

    %% Intermediate visualisation
    if mod(t,vis_step)==0
        fprintf('Iter:\t%d\tLoss:%.4f\tE_in:%.4f\tE_ex:%.4f\tR_disp:%.8f\n',t,L,E_in,E_ex,R_disp)

        figure(1)
        scatter(x(:,1)+u,x(:,2)+v,5,sqrt(u.^2+v.^2),'filled')
        t_str=sprintf('Iter: %d    Loss = %.4f', t, L);
        title(t_str,'Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
        axis equal
        colormap jet
        axis([0 50 0 75])
        xlabel('x (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
        ylabel('y (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
        set(gcf,'position',[0,400,220,300])
        drawnow

        figure(2)
        plot(L_hist(:,1),'LineWidth',2)
        axis([0 max_iter -2000 0])
        title('Loss history','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
        xlabel('Iteration','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
        ylabel('Overall loss functional','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
        set(gcf,'position',[0,100,380,200])
        drawnow
    end

    %% Conduct the Adam optimiser (gradient descendant algorithm)
    a=[a_u;a_v];
    L_grad=[L_grad_u; L_grad_v];

    m1=(1 - beta1) * L_grad + beta1 * m1;
    m2=(1 - beta2) * (L_grad.^2) + beta2 * m2;
    mhat = m1 / (1 - beta1^t);
    vhat = m2 / (1 - beta2^t);
    a = a - lr * mhat ./ (sqrt(vhat) + e);

    a_u=a(1:length(c),1);
    a_v=a(length(c)+1:length(c)*2,1);

end
walltime=toc;

%% Output data
fprintf('Final loss: %.4f\t CPU time: %.4f s\n',L,walltime)

if RBF_type==1
    save('output_Gauss.mat','a_u',"a_v","L_hist","walltime");
elseif RBF_type==2
    save('output_FifthPiecewise.mat','a_u',"a_v","L_hist","walltime");
end

% %% Visualisation
for i_s=1:length(x)
    
    % The deformation tensor
    F=[ux(i_s)+1 uy(i_s);
        vx(i_s) vy(i_s)+1];
    
    % The 1st Piola–Kirchhoff stress tensor
    P=(mu*(F-inv(F)')+la*(J(i_s)-1)*J(i_s)*inv(F)');
    % P=nu*F+(la*log(J(i_s))-nu)*inv(F)';
    
    % The Cauchy stress tensor
    sig=1/J(i_s)*P*F';
    % sig=(nu*F+(la*log(J(i_s))-nu)*inv(F)')*F';
    sig11(i_s)=sig(1,1);
    sig12(i_s)=sig(1,2);
    sig21(i_s)=sig(2,1);
    sig22(i_s)=sig(2,2);

    Mises(i_s)=sqrt(0.5*((sig11(i_s)-sig22(i_s))^2+3*sig12(i_s)^2));
end

figure(3)
subplot(2,3,1)
scatter(x(:,1),x(:,2),5,u,'filled');
axis equal
colorbar
colormap 'jet'
axis([0 50 0 62])
xlabel('x (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
ylabel('y (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
title('U (RPIM-NNS)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)

subplot(2,3,2)
scatter(x(:,1),x(:,2),5,v,'filled');
axis equal
colorbar
colormap 'jet'
axis([0 50 0 62])
xlabel('x (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
ylabel('y (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
title('V (RPIM-NNS)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)

subplot(2,3,4)
scatter(x(:,1),x(:,2),5,sig11,'filled');
axis equal
colorbar
colormap 'jet'
axis([0 50 0 62])
xlabel('x (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
ylabel('y (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
title('\sigma_{x} (RPIM-NNS)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)

subplot(2,3,5)
scatter(x(:,1),x(:,2),5,sig22,'filled');
axis equal
colorbar
colormap 'jet'
axis([0 50 0 62])
xlabel('x (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
ylabel('y (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
title('\sigma_{y} (RPIM-NNS)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)

subplot(2,3,6)
scatter(x(:,1),x(:,2),5,sig12,'filled');
axis equal
colorbar
colormap 'jet'
axis([0 50 0 62])
xlabel('x (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
ylabel('y (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
title('\tau_{xy} (RPIM-NNS)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)

set(gcf,'position',[400,100,950,550])

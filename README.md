# Radial point interpolation method empowered with neural network solvers (RPIM-NNS)

This repository provides MATLAB codes for the **RPIM-NNS**. 

The radial point interpolation method (RPIM) is a powerful meshless method for computational mechanics. Recently, deep learning (DL), especially the physics-informed neural network (PINN), has earned great attention for solving partial differential equations. We migrated the solver framework from DL and PINN to the traditional RPIM, attempting to combine the advantages of PINN as well as conventional computational mechanics. Therefore, we proposed an RPIM empowered by neural network solvers (RPIM-NNS). This repository provides the MATLAB codes of the proposed RPIM-NNS for 2D mechanics problems, including:

 - 2D cantilever beam with linear elastic material under the small deformation assumption (**Section 3.6** in the manuscript)
      
 - 2D Cook's membrane with neo-Hookean material under large deformation (**Section 4.1** in the manuscript)  
               
This paper has been accepted by **Computer Methods in Applied Mechanics and Engineering**. For more details in terms of implementations and more interesting numerical examples, please refer to our paper.

# Run code
Please run the file "RPIM_NNS_main.m" in each case. All the input data is prepared in "Coord.mat". Two radial basis functions (RBFs), namely the Gaussian type RBF and the fifth-order piece-wise RBF, are available. The output (results) are also saved in ".m" files.

# Paper link
Will be available soon.

# Enviornmental settings
 - MATLAB 2022b

# Cite as
[1] J. Bai, G.-R. Liu, T. Rabczuk, Y. Wang, X.-Q. Feng, Y.T. Gu, A robust radial point interpolation method empowered with neural network solvers (RPIM-NNS) for nonlinear solid mechanics, Computer Methods in Applied Mechanics and Engineering (2024). 

# Contact us
For questions regarding the code, please contact:

Dr. Jinshuai Bai: jinshuaibai@gmail.com or bjs@mail.tsinghua.edu.cn  
Prof. YuanTong Gu: yuantong.gu@qut.edu.au  

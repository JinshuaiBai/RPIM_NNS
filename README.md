# Radial point interpolation method empowered with neural network solvers (RPIM-NNS)

This repository provides MATLAB codes for the **RPIM-NNS**. 

<p align="justify">
The radial point interpolation method (RPIM) is a powerful meshless method for computational mechanics. Recently, deep learning (DL), especially the physics-informed neural network (PINN), has earned great attention for solving partial differential equations. We migrated the solver framework from DL and PINN to the traditional RPIM, attempting to combine the advantages of PINN as well as conventional computational mechanics. Therefore, we proposed an RPIM empowered by neural network solvers (RPIM-NNS). This repository provides the MATLAB codes of the proposed RPIM-NNS for 2D mechanics problems, including:
</p>

 - 2D cantilever beam with linear elastic material under the small deformation assumption (**Section 3.6** in the manuscript)
   ![Canti_results](https://github.com/JinshuaiBai/LSWR_loss_function_PINN/assets/103013182/256c25d2-3a9e-41cb-8369-c798ab387468)  
   <p align="center">
       <strong>Fig. 1.</strong> Results for the 2D cantilever beam problem.
   </p>
      
 - 2D Cook's membrane with neo-Hookean material under large deformation (**Section 4.1** in the manuscript)
   ![Cook_results](https://github.com/JinshuaiBai/LSWR_loss_function_PINN/assets/103013182/e545637e-a757-4024-aa8d-fb5a46d0ba2b)
   <p align="center">
       <strong>Fig. 2.</strong> Results for the 2D Cook's membrane problem.
   </p>
               
This paper has been accepted by **Computer Methods in Applied Mechanics and Engineering**. For more details in terms of implementations and more interesting numerical examples, please refer to our paper.

# Run code
<p align="justify">
Please run the file "RPIM_NNS_main.m" in each case. All the input data is prepared in "Coord.mat". Two radial basis functions (RBFs), namely the Gaussian type RBF and the fifth-order piece-wise RBF, are available. The output (results) are also saved in ".m" files.
</p>

# Paper link
Will be available soon.

# Enviornmental settings
 - MATLAB 2022b

# Cite as
<p align="justify">
[1] J. Bai, G.-R. Liu, T. Rabczuk, Y. Wang, X.-Q. Feng, Y.T. Gu, A robust radial point interpolation method empowered with neural network solvers (RPIM-NNS) for nonlinear solid mechanics, Computer Methods in Applied Mechanics and Engineering (2024). 
</p>

# Contact us
For questions regarding the code, please contact:

Dr. Jinshuai Bai: jinshuaibai@gmail.com or bjs@mail.tsinghua.edu.cn  
Prof. YuanTong Gu: yuantong.gu@qut.edu.au  

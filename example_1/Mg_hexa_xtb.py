import mlptrain as mlt
import numpy  as np
import ase.io as aio
from mlptrain.box import Box

mlt.Config.mace_params['calc_device'] = 'cuda'


## Beginning of the code
if __name__ == '__main__':
    
    system = mlt.System(mlt.Molecule('TPSSH.xyz',charge=2, mult=1),box = [100,100,100])
    
    mg = mlt.potentials.MACE('mg_aqua', system = system)
    
    mg.al_train(method_name = 'xtb', temp=400, 
                  max_active_iters = 10, 
                  max_active_time = 3000)

    trajectory_gas = mlt.md.run_mlp_md(configuration=system.random_configuration(),
                                   mlp = mg,
                                   fs=10000,
                                   temp=300,
                                   dt=0.5,
                                   interval=10)

    trajectory_gas.save(filename='mg_aqua_cluster_trajectory.xyz')
    trajectory_gas.compare(mg, 'xtb')
    trajectory_gas.save_xyz(filename='mg_aqua_cluster_trajectory_true.xyz',true=True, predicted = False)
    trajectory_gas.save_xyz(filename='mg_aqua_cluster_trajectory_predicted.xyz',true=False, predicted = True)  
  

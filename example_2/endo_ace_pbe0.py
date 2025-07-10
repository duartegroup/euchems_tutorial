import mlptrain as mlt

mlt.Config.n_cores = 10
mlt.Config.orca_keywords = ['PBE0', 'def2-SVP', 'EnGrad']
mlt.Config.mace_params['calc_device'] = 'cuda'

if __name__ == '__main__':
    system = mlt.System(
        mlt.Molecule('cis_endo_TS_PBE0.xyz', charge=0, mult=1), box=None
    )

    mace = mlt.potentials.MACE('endo_ace', system=system)

    mace.al_train(
        method_name='orca',
        temp=500,
        max_active_time=1000,
        fix_init_config=True,
        keep_al_trajs=True,
    )

    # Run some dynamics with the potential
    trajectory = mlt.md.run_mlp_md(
        configuration=system.configuration,
        mlp=mace,
        fs=200,
        temp=300,
        dt=0.5,
        interval=10,
    )

    # and compare, plotting a parity diagram and E_true, ∆E and ∆F
    
    trajectory.save(filename='mg_aqua_cluster_trajectory.xyz')
    trajectory.compare(mace, 'orca')

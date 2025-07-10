import mlptrain as mlt

mlt.Config.mace_params['calc_device'] = 'cuda'
mlt.Config.orca_keywords = ["PBE0", "def2-SVP", "EnGrad", "CPCM(Water)"]

if __name__ == "__main__":
    system = mlt.System(mlt.Molecule("ch3cl_f.xyz", charge=-1, mult=1), box=None)

    # Define CV and attach an upper wall
    #avg_r = mlt.PlumedAverageCV(name="avg_r", atom_groups=((0, 1), (0, 2)))

    #diff_r = mlt.PlumedDifferenceCV(name="diff_r", atom_groups=((0, 2), (0, 1)))

    r_cl = mlt.PlumedAverageCV(name="r_cl", atom_groups=(0, 2))
    r_cl.attach_upper_wall(location=5, kappa=100)

    r_f = mlt.PlumedAverageCV(name="r_f", atom_groups=(0, 1))
    r_f.attach_upper_wall(location=5, kappa=100)


    # Define the potential and train using WTMetaD AL (inherit_metad_bias=True)

    mace = mlt.potentials.MACE("r1_wtmetad_stagetwo_compiled", system=system)

    # Attach CVs to the metadynamics object
    metad = mlt.Metadynamics(cvs=(r_cl,r_f))

    # Generate a starting metadynamics configuration

    config = system.random_configuration()

    # run the optional function to help choosing parameters
    width = metad.estimate_width(configurations=config, mlp=mace, plot=True)

    # for width estimation

    metad.run_metadynamics(
        configuration=config,
        mlp=mace,
        temp=300,
        dt=0.5,
        ps=100,
        interval=10,
        n_runs=3,
        width=width,
        height=0.01,
        biasfactor=10,
    )

    metad.plot_fes()

    metad.plot_fes_convergence(stride=10, n_surfaces=5)

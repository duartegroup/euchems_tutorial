import mlptrain as mlt

mlt.Config.n_cores = 10
mlt.Config.orca_keywords = ["PBE0", "def2-SVP", "EnGrad", "CPCM(Water)"]

if __name__ == "__main__":
    system = mlt.System(mlt.Molecule("ch3cl_f.xyz", charge=-1, mult=1), box=None)

    # Define CV for WTMetaD AL (r_cl - r_f)

    diff_r = mlt.PlumedDifferenceCV(name="diff_r", atom_groups=((0, 2), (0, 1)))

    # Define CV and attach an upper wall
    
    avg_r = mlt.PlumedAverageCV(name="avg_r", atom_groups=((0, 1), (0, 2)))
    avg_r.attach_upper_wall(location=2.5, kappa=1000)

    # Initialise PlumedBias for WTMetaD AL
    bias = mlt.PlumedBias(cvs=(avg_r, diff_r))
    bias.initialise_for_metad_al(width=0.05, cvs=diff_r, biasfactor=100)

    # Define the potential and train using WTMetaD AL (inherit_metad_bias=True)

    mace = mlt.potentials.MACE("r1_wtmetad", system=system)
    mace.al_train(
        method_name="orca",
        temp=300,
        n_init_configs=5,
        n_configs_iter=5,
        max_active_iters=50,
        min_active_iters=30,
        inherit_metad_bias=True,
        bias=bias,
    )

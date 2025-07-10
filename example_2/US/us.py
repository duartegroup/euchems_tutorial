import mlptrain as mlt
import numpy as np
from mlptrain.box import Box
from mlptrain.log import logger

mlt.Config.mace_params["calc_device"] = "cuda"

if __name__ == "__main__":
    us = mlt.UmbrellaSampling(zeta_func=mlt.AverageDistance((1, 12), (6, 11)), kappa=10)
    temp = 300

    irc = mlt.ConfigurationSet()
    irc.load_xyz(filename="irc_IRC_Full_trj.xyz", charge=0, mult=1)

    for config in irc:
        config.box = Box([100, 100, 100])

    irc.reverse()

    TS_mol = mlt.Molecule(name="cis_endo_TS_PBE0.xyz", charge=0, mult=1, box=None)

    system = mlt.System(TS_mol, box=Box([100, 100, 100]))

    endo = mlt.potentials.MACE("endo_ace_stagetwo", system)

    us.run_umbrella_sampling(
        irc,
        mlp=endo,
        temp=temp,
        interval=5,
        dt=0.5,
        n_windows=15,
        init_ref=1.55,
        final_ref=4,
        ps=10,
    )
    us.save("wide_US")

    # Run a second, narrower US with a higher force constant
    us.kappa = 20
    us.run_umbrella_sampling(
        irc,
        mlp=endo,
        temp=temp,
        interval=5,
        dt=0.5,
        n_windows=15,
        init_ref=1.7,
        final_ref=2.5,
        ps=10,
    )

    us.save("narrow_US")

    total_us = mlt.UmbrellaSampling.from_folders("wide_US", "narrow_US", temp=temp)
    total_us.wham()

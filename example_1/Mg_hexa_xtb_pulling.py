import mlptrain as mlt
import numpy as np
import ase.io as aio

mlt.Config.mace_params["calc_device"] = "cuda"

if __name__ == "__main__":
    system = mlt.System(mlt.Molecule("TPSSH.xyz", charge=2, mult=1), box=None)
    mg = mlt.potentials.MACE("mg_aqua_stagetwo", system=system)
    al_bias = mlt.PlumedBias(filename="plumed.dat")

    trajectory_gas = mlt.md.run_mlp_md(
        configuration=system.random_configuration(),
        mlp=mg,
        fs=500,
        temp=300,
        dt=0.5,
        interval=25,
        bias=al_bias,
    )

    trajectory_gas.save(filename="mg_aqua_pulling_trajectory.xyz")
    trajectory_gas.compare(mg, "xtb")

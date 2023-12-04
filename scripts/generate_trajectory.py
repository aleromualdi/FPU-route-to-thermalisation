import argparse
import os
from pathlib import Path

import numpy as np

from src.integrator import FPUT_Integrator


def main(args):

    output_path = Path(os.getcwd()).parent / args.output_dir_name
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    fpu = FPUT_Integrator(
        num_atoms=args.num_atoms,
        num_modes=args.num_modes,
        initial_mode_number=args.initial_mode_number,
        initial_mode_amplitude=args.initial_mode_amp,
        t_step=args.time_step,
        t_max=args.tmax,
        alpha=args.alpha,
        beta=args.beta,
    )

    output = fpu.run(method="verlet")

    output_names = ["times", "q", "p", "mode_energies"]
    for x, name in zip(output, output_names):
        np.save(output_path + "/" + name, x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-atoms",
        type=int,
        help="number of particles in the system",
        default=32,
    )
    parser.add_argument(
        "--tmax",
        type=int,
        help="maximum time of simulation",
        default=1200000,
    )
    parser.add_argument(
        "--time-step",
        type=float,
        help="integration time step",
        default=0.05,
    )
    parser.add_argument(
        "--initial-mode-number",
        type=int,
        help="initial mode number",
        default=1,
    )
    parser.add_argument(
        "--num-modes",
        type=int,
        help="number of modes to be computed",
        default=10,
    )
    parser.add_argument(
        "--initial-mode-amp",
        type=int,
        help="initial mode amplitude",
        default=10.0,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="first order nonlinearity coef",
        default=0.0,
    )
    parser.add_argument(
        "--beta",
        type=float,
        help="second order nonlinearity coef",
        default=1.8,
    )
    parser.add_argument(
        "--output-dir-name",
        type=str,
        help="output directory name",
        default='data/fermi_1.8/',
    )

    args = parser.parse_args()
    main(args)

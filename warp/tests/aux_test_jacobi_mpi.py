# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the MPI Jacobi example's periodic halo exchange with three ranks and host buffers."""

import numpy as np
from mpi4py import MPI

from warp.examples.distributed.example_jacobi_mpi import Example, calc_decomp_1d


def main():
    comm = MPI.COMM_WORLD
    assert comm.size == 3, "Three ranks are required to distinguish the upper and lower neighbors."

    # Seven global rows split into three uneven slabs: [1, 2, 3], [4, 5], [6, 7].
    start, count = calc_decomp_1d(7, comm.rank, comm.size)
    example = Example.__new__(Example)
    example.mpi_comm = comm
    example.iy_start = 1
    example.iy_end = count + 1
    example.upper_neighbor = (comm.rank - 1) % comm.size
    example.lower_neighbor = (comm.rank + 1) % comm.size
    example.a_new = np.zeros((count + 2, 1), dtype=np.int32)
    example.a_new[1:-1, 0] = np.arange(start + 1, start + count + 1)

    example.apply_periodic_bc()

    # Check both halos and all owned rows, including the wrap from row 7 to row 1.
    expected = ([7, 1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 1])
    np.testing.assert_array_equal(example.a_new[:, 0], expected[comm.rank], err_msg=f"MPI rank {comm.rank}")


if __name__ == "__main__":
    main()

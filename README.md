# HITS and PageRank Implementations (MPI, OpenMP, CUDA)

This repository contains three implementations of the HITS and PageRank algorithms using:

- **MPI** (`hits_pagerank_mpi_code.c`)
- **OpenMP** (`hits_pagerank_omp.c`)
- **CUDA** (`cuda_hits_pagerank.cu`)

---

##  Requirements

Before you compile and run the programs, ensure you have:

- **MPI compiler** (`mpicc`) — e.g., from OpenMPI or MPICH
- **OpenMP-capable compiler** (`gcc` with `-fopenmp` support)
- **CUDA toolkit** (`nvcc`)

You can check if they’re installed:
```bash
mpicc --version
gcc --version
nvcc --version
```

---

## Building the Programs

A `Makefile` is provided.

To compile all executables, simply run:
```bash make
```

This will generate:
- `mpi` → MPI implementation
- `omp` → OpenMP implementation
- `cuda` → CUDA implementation

To clean (delete) the compiled binaries:
```bash
make clean
```

---

## Running the Programs

### MPI
Run with `mpirun` or `mpiexec`:
```bash
mpirun -np <num_processes> ./mpi <input_file>
```

Example:
```bash
mpirun -np 4 ./mpi data/Web-Google.mtx
```

---

### OpenMP
Run directly (you can control threads using the `OMP_NUM_THREADS` environment variable):
```bash
export OMP_NUM_THREADS=<num_threads>
./omp <input_file>
```

Example:
```bash
export OMP_NUM_THREADS=8
./omp data/Web-Google.mtx
```

---

### CUDA
Run directly on a machine with a compatible GPU:
```bash
./cuda <input_file>
```

Example:
```bash
./cuda data/Web-Google.mtx
```

---

## 📂 Input Files

As the input files are bigger in size, they have been added to the drivelink mentioned below:
https://drive.google.com/drive/folders/1bvuaJSGhqeYtiz8RmXBc4zc1oirNND9y?usp=drive_link

Make sure the `.mtx` (Matrix Market) input files are in the correct path or adjust the command to provide the full path.

Example input files:
- `Web-Google.mtx`
- `ca-GrQc.mtx`
- `Orkut.mtx`
- `Wiki-Talk.mtx`

---

## Notes

- Ensure you have the necessary permissions to execute the binaries (`chmod +x` if needed).
- For CUDA, make sure your system has a supported NVIDIA GPU and drivers.
- If you encounter errors, check that your environment has the required compilers and libraries installed.
- The output for the last run iteration of the programs are in the below mentioned drive link:
  https://drive.google.com/drive/folders/1bvuaJSGhqeYtiz8RmXBc4zc1oirNND9y?usp=drive_link

---

## Contact

If you have issues or questions, feel free to reach out to the author. 

Contributors:
Sandhiya Shunmugavel (sshunmu2@ncsu.edu)
Sanjaey Shunmuga Sundaram (sshunmu@ncsu.edu)

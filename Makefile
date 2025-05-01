# Compiler settings
MPICC = mpicc
OMPCC = gcc -fopenmp
NVCC  = nvcc

# Source files
MPI_SRC   = hits_pagerank_mpi_code.c
OMP_SRC   = hits_pagerank_omp.c
CUDA_SRC  = cuda_hits_pagerank.cu

# Output executables
MPI_EXE   = mpi
OMP_EXE   = omp
CUDA_EXE  = cuda

# Compiler flags
CFLAGS    = -O2
NVFLAGS   = -O2

# Default target: build all
all: $(MPI_EXE) $(OMP_EXE) $(CUDA_EXE)

# MPI build
$(MPI_EXE): $(MPI_SRC)
	$(MPICC) $(CFLAGS) -o $(MPI_EXE) $(MPI_SRC)

# OpenMP build
$(OMP_EXE): $(OMP_SRC)
	$(OMPCC) $(CFLAGS) -o $(OMP_EXE) $(OMP_SRC)

# CUDA build
$(CUDA_EXE): $(CUDA_SRC)
	$(NVCC) $(NVFLAGS) -o $(CUDA_EXE) $(CUDA_SRC)

# Clean up
clean:
	rm -f $(MPI_EXE) $(OMP_EXE) $(CUDA_EXE)

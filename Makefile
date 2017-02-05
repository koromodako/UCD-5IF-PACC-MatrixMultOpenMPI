
all: build

build:
	mpicc -o matmultmpi_1d main_1d.c
	mpicc -o matmultmpi_2d main_2d.c

clean:
	rm -f matmultmpi_*

run:
	mpirun -np 4 --hostfile hostfile matmultmpi_${V} ${N}

benchmark:
	mpirun -np 4 --hostfile hostfile matmultmpi_${V} 4
	mpirun -np 4 --hostfile hostfile matmultmpi_${V} 8
	mpirun -np 4 --hostfile hostfile matmultmpi_${V} 16
	mpirun -np 4 --hostfile hostfile matmultmpi_${V} 32
	mpirun -np 4 --hostfile hostfile matmultmpi_${V} 64
	mpirun -np 4 --hostfile hostfile matmultmpi_${V} 128
	mpirun -np 4 --hostfile hostfile matmultmpi_${V} 256
	mpirun -np 4 --hostfile hostfile matmultmpi_${V} 512
	mpirun -np 4 --hostfile hostfile matmultmpi_${V} 1024
	mpirun -np 4 --hostfile hostfile matmultmpi_${V} 2048

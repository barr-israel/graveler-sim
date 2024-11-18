# Graveler Simulation
Simulating graveler battles for the softlock described in [Pikasprey‬'s](https://www.youtube.com/watch?v=GgMl4PrdQeo&t=0s) and [Shoddycast's](https://www.youtube.com/watch?v=M8C8dHQE2Ro) videos

## Download
Simply download a version matching your machine from releases, but performance will likely be lower due to not having native target compilation, and will be set using half the available threads.  
A different executable was compiled for different x86_64 instruction sets([wikipedia](https://en.wikipedia.org/wiki/X86-64#Microarchitecture)), newer sets will have a better performance, but an executable for a newer set than available on the machine will not run.
## Compilation
To compile the code, you need to have rust nightly installed, and then run the performance maximised build command:
```rust
cargo +nightly build --profile max
```
+nightly is optional if nightly is the default toolchain on the machine.  
native targetting was enabled for all tier 1 64 bit targets, for other targets adding the -Ctarget-cpu=native to the RUSTFLAGS will increase performance significantly.  
the amount of threads used optimally can be different between different CPUs, it is set to half the available threads by default and can be changed directly in the code if testing other values is desired.

The executable will be generated in ./target/max/graveler (on windows it will be graveler.exe)

## Usage
Simple run the executable via any terminal and the highest number found will be printed shortly.

## Performance
Performance was measured using hyperfine
| CPU                               | Single Thread | Half Threads | All Threads |
|-----------------------------------|---------------|--------------|-------------|
| i7-10750H 6 Cores 12 Threads      | 2.78s         | 512ms        | 531ms       |
| Ryzen 7950X3D 16 Cores 32 Threads | 1.78s         | 134ms        | 117ms       |

## GPU Implementation
 In the cuda folder, there is a CUDA implementation of a nearly identical algorithm, for running on an Nvidia GPU.  
 To compile it, you need CUDA installed, and can simply run `make` to compile both the normal version and the benchmark version.  
 The benchmark version runs the kernel 50 times as warmp up and then 1000 more to time it and output the average of the 1000 runs.  
 The output is only for the kernel time and summarizing the kernel results, it does not include the CUDA runtime initalization, which can take significantly longer then the kernel.  
| GP U                  | Average |
|-----------------------|---------|
| RTX 2070 Mobile Max-Q | 31.51ms |
| RTX 4080              | 6.36ms  |

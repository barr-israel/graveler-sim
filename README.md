# Graveler Simulation
Simulating graveler battles for the softlock described in [Pikasprey‬'s](https://www.youtube.com/watch?v=GgMl4PrdQeo&t=0s) and [Shoddycast's](https://www.youtube.com/watch?v=M8C8dHQE2Ro) videos

## Download
Simply download a version matching your machine from releases, but performance will likely be lower due to not having native target compilation, and will be set using half the available threads.  
A different executable was compiled for different x86_64 instruction sets([wikipedia](https://en.wikipedia.org/wiki/X86-64#Microarchitecture)), an executable for a newer set than available on the machine will not run.
## Compilation
To compile the code, you need to have rust nightly installed, and then run the performance maximised build command:
```rust
cargo +nightly build --profile max
```
+nightly is optional if nightly is the default toolchain on the machine.  
native targetting was enabled for all tier 1 64 bit targets, for other targets adding the -Ctarget-cpu=native will increase performance significantly.  
the amount of threads used optimally can be different between different CPUs, it is set to half the available threads by default and can be changed directly in the code if testing other values is desired.

The executable will be generated in ./target/max/graveler (on windows it will be graveler.exe)

## Usage
Simple run the executable via any terminal and the highest number found will be printed shortly.

## Performance
Performance was measured using hyperfine
| CPU                               | Single Thread | Half Threads | All Threads |
|-----------------------------------|---------------|--------------|-------------|
| i7-10750H 6 Cores 12 Threads      | 2.78s         | 512ms        | 531ms       |
| Ryzen 7950x3d 16 Cores 32 Threads | 1.78s         | 134ms        | 117ms       |

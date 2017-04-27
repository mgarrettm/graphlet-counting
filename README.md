# graphlet-counting
Graphlet counting implementation in CUDA

## Running on CRC cluster. 
Not sure how much larger the graphs will get, but, there is interactive access to the 2 main CRC GPU machines. If the jobs don't run long, then this will probably be enough.

### General procedure for compiling and running (without submitting to the queue).

Log into one of the CRC frontend machines: netid@crc*.crc.nd.edu
(crcfe01.crc.nd.edu, crcfe02.crc.nd.edu, crcfeIB01.crc.nd.edu)

To get the interactive shell, run the following command for the NVIDIA GeForce GTX 1080 GPU
```bash
[snjoroge@crcfe01 ~]$ qrsh -q gpu-debug
```
or, for the NVIDIA GeForce GTX TITAN X GPU

```bash
[snjoroge@crcfe01 ~]$ qrsh -q gpu
```
I don't think you can log into the GPU machines directly. I tried and failed. Prompt will change to the machine you are on.

Then, load the cuda compiler (Default is version 7.5)
```bash
[snjoroge@qa-titanx-001]$ module load cuda
```

Locate your code in your AFS space. Compile with the follow command.
```bash
[snjoroge@qa-titanx-001 graphlet-counting]$ nvcc -o graphlets graphlets.cu
```

Run the program with a graph
```bash
[snjoroge@qa-titanx-001 graphlet-counting]$ ./graphlets data/network.txt
```

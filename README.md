# GPUProject
Reverse engineering the memory hierarchy of a GPU.

Use nvcc pChase pChase.cu to compile.

Use nvcc -Xptxas -dlcm=cg -o pChase pChase.cu to compile with L1 cache disabled.
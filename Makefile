all: pChase pChase_noL1 devicequery
pChase: pChase.cu
	nvcc pChase.cu -o pChase
pChase_noL1: pChase.cu
	nvcc -Xptxas -dlcm=cg pChase.cu -o pChase_noL1
devicequery: devicequery.cu
	nvcc devicequery.cu -o devicequery

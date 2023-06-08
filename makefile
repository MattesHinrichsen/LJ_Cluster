all:
	g++ -o main.exe main.cpp -Ofast -march=native -I ../Eigen -fopenmp
	.\main.exe
debug:
	g++ -o main.exe main.cpp -O1 -march=native -I ../Eigen -fopenmp -g -ggdb -funsafe-math-optimizations
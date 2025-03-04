# UCI chess engine

Estimated strength 3400 Elo (Blitz)

## efficiently updatable neural network (NNUE) evaluation
- 16 king buckets (total 2x16x768 inputs)
- 256x2 -> 32 -> 1 architecture
- clipped relu activation function
- quantized 16-bit fixed point arithmetics
- vectorized for AVX2
- trained on selfplay positions (~4 billion positions)
- CPU trainer (backpropagation + ADAM optimizer)

## principal variation search

- iterative deepening
- aspiration window
- transposition table
- evaluation cache
- bitboards (PEXT sliding piece movegen)
- multi pv support
- multithreading support (lazy smp)
- improving heurestic

- pruning

	- null move pruning
	- reverse futility pruning
	- late move pruning
	- futility pruning
	- history pruning
	- SEE pruning for captures
	- multicut (based on singular search)
	- mate distance pruning

- reductions

	- late move reduction
	- history based reductions
	- internal iterative reductions

- move ordering

	- hash move
	- static exchange evaluation (SEE)
	- history heurestic
	- capture history
	- continuation history
	- killer heurestic
	- counter move heurestic
	- threats

- static evaluation correction history (pawn structure and material configuration)
- singular extension

## quiscence search

- TT probing
- SEE move ordering
- SEE pruning	

## requirements

x86-64 cpu with AVX2 and BMI2 extensions (Intel Haswell or newer. AMD Excavator or newer)

	

module GlobalRef

using Debugger

using PyPlot: figure, close, plot, show
using DP
using GridWorld
using ProbUtils

macro dispSize(P)
	return :( println(size($P)) )
end

macro dispMatrix(P)
	return :( display($P); println())
end

Dkl(P, Q, dims) = sum( log.((P ./ Q ) .^P), dims=dims )

pgxi_xj(Pgxiai_xj, Pgxi_ai) = ( sum(Pgxiai_xj .* Pgxi_ai, dims=4) )

pxi_xj(Pgxiai_xj, Pgxi_ai, Pg) = ( sum(Pgxiai_xj .* Pgxi_ai .* Pg, dims=[1, 4]) )

pgxi(Pgxi_xj, g) = ((Pgxi_xj[g, 1, :, 1, :, 1] ^ 20)[1, :]) 

pxi(Pxi_xj) = ((Pxi_xj[1, 1, :, 1, :, 1] ^ 200)[1, :]) 

pxi_ai(Pgxi_ai, Pg)   = ( sum(Pgxi_ai .* Pg, dims=1) )

pgh(Pg_h, Pg) = Pg_h .* Pg

ph(Pg_h, Pg) = (Pgh = pgh(Pg_h, Pg); res = sum(Pgh, dims=1); res)

phxi_ai(Pgxi_ai, Ph_g) = sum(Pgxi_ai .* Ph_g, dims=1 )  

ph_g(Pg_h, Pg, Ph) = ( Pg .* Pg_h ./ Ph )

pg_ai(Pgxi_ai, Pxi) = sum(Pgxi_ai .* Pxi, dims=3)

ph_ai(Phxi_ai, Pxi) = sum(Phxi_ai .* Pxi, dims=3)

phxi_ai(Pgxi_ai, Ph_g) = sum(Pgxi_ai .* Ph_g, dims=1)


pg_h(Ph, Pxi, Pgxi_ai, Phxi_ai, Pg_ai, Ph_ai, beta) = 
(
	P = Ph .* exp.(beta*(sum(Pxi .* Dkl(Pgxi_ai, Phxi_ai, (4,)), dims=3) - Dkl(Pg_ai, Ph_ai, (4,))));
	P ./ sum(P, dims=2)
)

Igh(Pgh, Ph, Pg) =  sum( log.(( Pgh ./ (Ph .* Pg)) .^ Pgh ) )
Ih_xiai(Phxiai, Ph_ai, Phxi_ai)= (Ph_xiai = Phxiai ./ sum(Phxiai, dims=[3, 4]); sum( (Phxi_ai ./ Ph_ai) .^ Phxiai))
L(Pgh, Ph, Pg, Phxiai, Ph_ai, Phxi_ai, beta) = Igh(Pgh, Ph, Pg) - beta*Ih_xiai(Phxiai, Ph_ai, Phxi_ai)

function goalIB(dims, Pgxi_ai, Pg, Pxi, BETA::StepRangeLen)

	I1 = []
	I2 = []
	L  = []
	for beta in BETA
		println(beta)
		L_tmp  = zeros(5)
		I1_tmp = zeros(5)
		I2_tmp = zeros(5)
		for i in 1:5
			Pg_h, Ph, Phxi_ai = goalIB(dims, Pgxi_ai, Pg, Pxi, beta)
			Pgh = pgh(Pg_h, Pg)
			Ph_ai = ph_ai(Phxi_ai, Pxi)
			Phxiai = Phxi_ai .* Ph_ai .* Pxi

			i1 = Igh(Pgh, Ph, Pg) 
			i2 = Ih_xiai(Phxiai, Ph_ai, Phxi_ai) 

# 	println((size(i1), size(i2)))
# 	display(i1); println()
# 	display(i2); println()



			l  = Igh(Pgh, Ph, Pg) - beta*Ih_xiai(Phxiai, Ph_ai, Phxi_ai)

			L_tmp[i]  = l
			I1_tmp[i] = i1
			I2_tmp[i] = i2
		end
		val, idx = findmin(L_tmp)
		push!(I1, I1_tmp[idx])
		push!(I2, I2_tmp[idx])
		push!( L,  L_tmp[idx])
	end

	return I1, I2, L

end

function goalIB(dims, Pgxi_ai, Pg, Pxi, beta::Float64)

	dimG, dimH =  9, 9

	Pg_h    = Normalize(rand(dimG, dimH, 1, 1, 1, 1), (2, ))

	Pg_ai   = pg_ai(Pgxi_ai, Pxi)

	Ph = []
	Phxi_ai = []

	for i in 1:5
		@bp
		Ph      = ph(Pg_h, Pg)
		Ph_g    = ph_g(Pg_h, Pg, Ph) 
		Phxi_ai = phxi_ai(Pgxi_ai, Ph_g)
		Ph_ai   = ph_ai(Phxi_ai, Pxi)
		Pg_h 	= pg_h(Ph, Pxi, Pgxi_ai, Phxi_ai, Pg_ai, Ph_ai, beta) 
	end

	return Pg_h, Ph, Phxi_ai
end

function main()

	dimX1, dimX2, dimA = (3, 3, 5)
	dimX = dimX1*dimX2
	dimG = dimX
	dimH = dimX
	dims = (dimX1, dimX2, dimA, dimX, 1)

	Pgxi_ai  = Array{Float64}(undef, dimG,    1, dimX, dimA,    1, 1)
	Phxi_ai  = Array{Float64}(undef,    1, dimH, dimX, dimA,    1, 1)
	Pgxiai_xj= Array{Float64}(undef, dimG,    1, dimX, dimA, dimX, 1)
	Pg       = Array{Float64}(undef, dimG,    1,    1,    1,    1, 1)
	Ph       = Array{Float64}(undef,    1, dimH,    1,    1,    1, 1)
	Ph_g     = Array{Float64}(undef, dimG, dimH,    1,    1,    1, 1)
	Pg_h     = Array{Float64}(undef, dimG, dimH,    1,    1,    1, 1)
	Pxi      = Array{Float64}(undef,    1,    1, dimX,    1,    1, 1)

	C    = CreateWalls(dims, "B")
	G    = GenerateGoalSet(dims, "fullX", C)
	Vgx, _Pgxi_ai, _Pgxiai_xj = SolveMazeGoalCond(dims, C, G)
	Pg[:, 1, 1, 1, 1, 1] = ones(dimG, 1)/dimG

	Pgxiai_xj[:, 1, :, :, :, 1] = _Pgxiai_xj
	Pgxi_ai[  :, 1, :, :, 1, 1] = _Pgxi_ai 	

	display( reshape(Vgx[7, :, 1, 1], (dimX1, dimX2)) ); println()

	Pgxi_xj = pgxi_xj(Pgxiai_xj, Pgxi_ai)

	Pxi_xj = pxi_xj(Pgxiai_xj, Pgxi_ai, Pg)


	Pg_h    = Normalize(rand(dimG, dimH, 1, 1, 1, 1), (2, ))
	Ph      = ph(Pg_h, Pg)
	Ph_g    = ph_g(Pg_h, Pg, Ph) 
	Phxi_ai = phxi_ai(Pgxi_ai, Ph_g)

	Pg_ai   = pg_ai(Pgxi_ai, Pxi)
	Ph_ai   = ph_ai(Phxi_ai, Pxi)

	println(size(Ph_ai))
	println(size(Pg_ai))

	Pg_h, Ph, Phxi_ai = goalIB(dims, Pgxi_ai, Pg, Pxi, 0.001)

# 	display(reshape(Ph, dimX1, dimX2)); println()
	display(reshape(Ph[1, :, 1, 1, 1, 1], (dimX1, dimX2) )); println()
	display(reshape(Pg[:, 1, 1, 1, 1, 1], (dimX1, dimX2) )); println()

	BETA = 10:-0.5:0.0000001
	Igh, Ih_xiai, L = goalIB(dims, Pgxi_ai, Pg, Pxi, 1.0)

	display(Igh); println()
	display(Ih_xiai); println()
# 	figure;
# 	plot(Igh, Ih_xiai)
# 	show()
# 	close()
end

# main()

end#module

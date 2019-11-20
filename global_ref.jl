module GlobalRef

# using PyPlot: figure, close, plot, show
using __DP
using GridWorld
using ProbUtils

idxG, idxH, idxXi, idxAi, idxXj, idxAj = (1, 2, 3, 4, 5, 6)

macro dispMatrix(P)
	return :( display($P); println())
end

Dkl(P, Q, dims) = sum( log.((P ./ Q ) .^P), dims=dims )

pgxi_xj(Pgxiai_xj, Pgxi_ai) = ( sum(Pgxiai_xj .* Pgxi_ai, dims=4) )

pxi_xj(Pgxiai_xj, Pgxi_ai, Pg) = ( sum(Pgxiai_xj .* Pgxi_ai .* Pg, dims=[idxG, idxAi]) )

pgxi(Pgxi_xj, g) = ((Pgxi_xj[g, 1, :, 1, :, 1] ^ 20)[1, :]) 

pxi(Pxi_xj) = ((Pxi_xj[1, 1, :, 1, :, 1] ^ 200)[1, :]) 

pxi_ai(Pgxi_ai, Pg)   = ( sum(Pgxi_ai .* Pg, dims=1) )

pgh(Pg_h, Pg) = Pg_h .* Pg

ph(Pg_h, Pg) = (Pgh = pgh(Pg_h, Pg); res = sum(Pgh, dims=idxG); res)

phxi_ai(Pgxi_ai, Ph_g) = sum(Pgxi_ai .* Ph_g, dims=idxG)  

ph_g(Pg_h, Pg, Ph) = ( Pg .* Pg_h ./ Ph )

pg_ai(Pgxi_ai, Pxi) = sum(Pgxi_ai .* Pxi, dims=idxXi)

ph_ai(Phxi_ai, Pxi) = sum(Phxi_ai .* Pxi, dims=idxXi)

function pg_h(Ph, Pxi, Pgxi_ai, Phxi_ai, Pg_ai, Ph_ai, beta)

	Dkl1 = Dkl(Pgxi_ai, Phxi_ai, (idxAi,))

	Dkl2 = Dkl(Pg_ai, Ph_ai, (idxAi,))

	P = Ph .* (exp.(beta*(sum(Pxi .* Dkl1, dims=idxXi) - Dkl2)));

	P ./ sum(P, dims=idxH)
end

Igh(Pgh, Ph, Pg) =  sum( log.(( Pgh ./ (Ph .* Pg)) .^ Pgh ) )

Ih_xiai(Phxiai, Ph_ai, Phxi_ai)= (Ph_xiai = Phxiai ./ sum(Phxiai, dims=[3, 4]); sum(log.((Phxi_ai ./ Ph_ai) .^ Phxiai)))

L(Pgh, Ph, Pg, Phxiai, Ph_ai, Phxi_ai, beta) = Igh(Pgh, Ph, Pg) - beta*Ih_xiai(Phxiai, Ph_ai, Phxi_ai)

pgxi_ai_update(Pxi_ai, beta, Qg) = ( Pgxi_ai = Pxi_ai .* exp.(beta*Qg); Pgxi_ai ./ sum(Pgxi_ai, dims=idxAi) ) 

function pgxi_ai_state( dims, Pg::Array{Float64}, Qg::Array{Float64}, Pgxiai_xj::Array{Float64}, beta::Float64 )

	dimX1, dimX2, dimA, dimG, dimH = dims
	dimX = dimX1*dimX2
	#here dimH equals 1, as no compression yet


	Pgxi_ai = Normalize( rand(dimG, 1, dimX, dimA, 1, 1), (idxAi,) )

	Pxi_xj  = pxi_xj(Pgxiai_xj, Pgxi_ai, Pg)

	Pxi     = pxi(Pxi_xj)

	Pg_ai   = sum( Pgxi_ai .* Pxi, dims=idxXi ) 	

	for i in 1:200

		Pgxi_ai = pgxi_ai_update(Pg_ai, beta, Qg)

		Pxi_xj  = pxi_xj(Pgxiai_xj, Pgxi_ai, Pg)

		Pxi     = pxi(Pxi_xj)

		Pg_ai   = sum( Pgxi_ai .* Pxi, dims=idxXi ) 	
	end

	return Pgxi_ai, Pg_ai, Pxi
end
function pgxi_ai_goal( dims, Pg::Array{Float64}, Qg::Array{Float64}, Pgxiai_xj::Array{Float64}, beta::Float64 )

	dimX1, dimX2, dimA, dimG, dimH = dims
	dimX = dimX1*dimX2
	#here dimH equals 1, as no compression yet


	Pgxi_ai = Normalize( rand(dimG, 1, dimX, dimA, 1, 1), (idxAi,) )

	Pxi_xj  = pxi_xj(Pgxiai_xj, Pgxi_ai, Pg)

	Pxi     = pxi(Pxi_xj)

	Pxi_ai   = sum( Pgxi_ai .* Pg, dims=idxG ) 	

	for i in 1:200

		Pgxi_ai = pgxi_ai_update(Pxi_ai, beta, Qg)

		Pxi_xj  = pxi_xj(Pgxiai_xj, Pgxi_ai, Pg)

		Pxi     = pxi(Pxi_xj)

		Pxi_ai   = sum( Pgxi_ai .* Pg, dims=idxG ) 	
	end

	return Pgxi_ai, Pxi_ai, Pxi
end


function goalIB(dims, Pg, Pgxi_ai, Pg_ai, Pxi, beta::Float64)

	dimX1, dimX2, dimA, dimG, dimH  = dims

	dimX = dimX1*dimX2

	Pg_h    	= Normalize(rand(dimG, dimH, 1, 1, 1, 1), (idxH, ))

	Ph      	= ph(Pg_h, Pg)

	Ph_g    	= ph_g(Pg_h, Pg, Ph) 

	Phxi_ai 	= phxi_ai(Pgxi_ai, Ph_g)

	Ph_ai   	= sum(Pg_ai .* Ph_g, dims=idxG)

	Phxiai 		= Phxi_ai .* Ph .* Pxi

	for i in 1:20

		Pg_h 		= pg_h(Ph, Pxi, Pgxi_ai, Phxi_ai, Pg_ai, Ph_ai, beta) 

		Ph      	= ph(Pg_h, Pg)

		Ph_g    	= ph_g(Pg_h, Pg, Ph) 

		Phxi_ai 	= phxi_ai(Pgxi_ai, Ph_g)

		Ph_ai   	= sum(Pg_ai .* Ph_g, dims=idxG)

		Phxiai 		= Phxi_ai .* Ph .* Pxi

		Pgh 	= pgh(Pg_h, Pg)

		i1 	= Igh(Pgh, Ph, Pg) 

		i2 	= Ih_xiai(Phxiai, Ph_ai, Phxi_ai) 


		println( (i, i1, i2) )
if i in (1, 20)
		for gi in 1:dimG
			println("gi=$(gi)")
			display(reshape(Pg_h[gi, :, 1, 1, 1, 1], (dimX1, dimX2))); println()
		end
# 		display(reshape(Ph[1, :, 1, 1, 1, 1], (dimX1, dimX2))); println()
end
	end

	return Pg_h, Ph, Phxi_ai

end


function main()

	dimX1, dimX2, dimA = (10, 10, 5)
	dimX = dimX1*dimX2
	dimG = -1
	dimH = -1

	dims = (dimX1, dimX2, dimA, dimG, dimH)

	C    = CreateWalls(dims, "GlobalRef5x5r9")
	G    = GenerateGoalSet(dims, "fullX", C)

	dimG = length(G)
	dimH = dimG
		
	dims = (dimX1, dimX2, dimA, dimG, dimH)

	Pgxi_ai  = Array{Float64}(undef, dimG,    1, dimX, dimA,    1, 1)
	Phxi_ai  = Array{Float64}(undef,    1, dimH, dimX, dimA,    1, 1)
	Pgxiai_xj= Array{Float64}(undef, dimG,    1, dimX, dimA, dimX, 1)
	Pg       = Array{Float64}(undef, dimG,    1,    1,    1,    1, 1)
	Ph       = Array{Float64}(undef,    1, dimH,    1,    1,    1, 1)
	Ph_g     = Array{Float64}(undef, dimG, dimH,    1,    1,    1, 1)
	Pg_h     = Array{Float64}(undef, dimG, dimH,    1,    1,    1, 1)
	Pxi      = Array{Float64}(undef,    1,    1, dimX,    1,    1, 1)

	Vgx, Qgxa, _Pgxi_ai, _Pgxiai_xj = SolveMazeGoalCond(dims, C, G)
	Pgxiai_xj[:, 1, :, :, :, 1] = _Pgxiai_xj
	Pgxi_ai[  :, 1, :, :, 1, 1] = _Pgxi_ai 	
	Pg      = Normalize( ones(dimG, 1, 1, 1, 1, 1), (idxG,) )

	beta = 10.0
	Pgxi_ai, Pxi_ai, Pxi = pgxi_ai_goal( dims, Pg, Qgxa, Pgxiai_xj, beta )

	# 	display(reshape(Vgx[1, :, :, :, :, :], (dimX1, dimX2))); println()
	# 	display(reshape(Qgxa[1, 1, :, :, 1, 1], (dimX, dimA))); println()
	display(reshape(Pxi[:, :, :, :, :, :], (dimX1, dimX2))); println()

	Pg_ai = sum( Pgxi_ai .* Pxi, dims=idxXi )

	goalIB(dims, Pg, Pgxi_ai, Pg_ai, Pxi, 100.0)

end

main()

end

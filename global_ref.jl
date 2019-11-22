module GlobalRef

using PyPlot: figure, close, plot, show, imshow, colorbar, subplot, subplots, savefig
using __DP
using GridWorld
using ProbUtils
using CuArrays
using CuArrays: cu

idxG, idxH, idxXi, idxAi, idxXj, idxAj = (1, 2, 3, 4, 5, 6)

macro dispMatrix(P)
	return :( display($P); println())
end


Dkl(P, Q, dims) = sum( log.((P ./ Q ) .^P), dims=dims )

pgxi_xj(Pgxiai_xj, Pgxi_ai) = ( sum(Pgxiai_xj .* Pgxi_ai, dims=4) )

pxi_xj(Pgxiai_xj, Pgxi_ai, Pg) = ( sum(Pgxiai_xj .* Pgxi_ai .* Pg, dims=[idxG, idxAi]) )

pgxi(Pgxi_xj, g) = ((Pgxi_xj[g, 1, :, 1, :, 1] ^ 20)[1, :]) 

pxi(Pxi_xj) = (reshape((Pxi_xj[1, 1, :, 1, :, 1] ^ 200)[1, :], (1, 1, size(Pxi_xj)[3], 1, 1, 1)))

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

function pgxi_ai_goal( dims, Pg::CuArray{BigFloat}, Qg::CuArray{BigFloat}, Pgxiai_xj::CuArray{BigFloat}, beta::BigFloat )

	dimX1, dimX2, dimA, dimG, dimH = dims
	dimX = dimX1*dimX2
	#here dimH equals 1, as no compression yet


	Pgxi_ai = Normalize( rand(eltype(Pg), dimG, 1, dimX, dimA, 1, 1), (idxAi,) )

	Pxi_xj  = pxi_xj(Pgxiai_xj, Pgxi_ai, Pg)

	Pxi     = pxi(Pxi_xj)

	Pxi_ai   = sum( Pgxi_ai .* Pg, dims=idxG ) 	

	for i in 1:5
		println("CUBigFloat ", i)

		Pgxi_ai = pgxi_ai_update(Pxi_ai, beta, Qg)

		Pxi_xj  = pxi_xj(Pgxiai_xj, Pgxi_ai, Pg)

		Pxi     = pxi(Pxi_xj)

		Pxi_ai   = sum( Pgxi_ai .* Pg, dims=idxG ) 	
	end

	return Pgxi_ai, Pxi_ai, Pxi
end

function pgxi_ai_goal( dims, Pg::Array{BigFloat}, Qg::Array{BigFloat}, Pgxiai_xj::Array{BigFloat}, beta::BigFloat )

	dimX1, dimX2, dimA, dimG, dimH = dims
	dimX = dimX1*dimX2
	#here dimH equals 1, as no compression yet


	Pgxi_ai = Normalize( rand(eltype(Pg), dimG, 1, dimX, dimA, 1, 1), (idxAi,) )

	Pxi_xj  = pxi_xj(Pgxiai_xj, Pgxi_ai, Pg)

	Pxi     = pxi(Pxi_xj)

	Pxi_ai   = sum( Pgxi_ai .* Pg, dims=idxG ) 	

	for i in 1:5
		println("BigFloat ", i)

		Pgxi_ai = pgxi_ai_update(Pxi_ai, beta, Qg)

		Pxi_xj  = pxi_xj(Pgxiai_xj, Pgxi_ai, Pg)

		Pxi     = pxi(Pxi_xj)

		Pxi_ai   = sum( Pgxi_ai .* Pg, dims=idxG ) 	
	end

	return Pgxi_ai, Pxi_ai, Pxi
end

function pgxi_ai_goal( dims, Pg::Array{Float64}, Qg::Array{Float64}, Pgxiai_xj::Array{Float64}, beta::Float64 )

	dimX1, dimX2, dimA, dimG, dimH = dims
	dimX = dimX1*dimX2
	#here dimH equals 1, as no compression yet


	Pgxi_ai = Normalize( rand(dimG, 1, dimX, dimA, 1, 1), (idxAi,) )

	Pxi_xj  = pxi_xj(Pgxiai_xj, Pgxi_ai, Pg)

	Pxi     = pxi(Pxi_xj)

	Pxi_ai   = sum( Pgxi_ai .* Pg, dims=idxG ) 	

	for i in 1:5
println("Float64 ", i)

		Pgxi_ai = pgxi_ai_update(Pxi_ai, beta, Qg)

		Pxi_xj  = pxi_xj(Pgxiai_xj, Pgxi_ai, Pg)

		Pxi     = pxi(Pxi_xj)

		Pxi_ai   = sum( Pgxi_ai .* Pg, dims=idxG ) 	
	end

	return Pgxi_ai, Pxi_ai, Pxi
end


function goalIB(dims, Pg, Pgxi_ai, Pg_ai, Pxi, beta)

	dimX1, dimX2, dimA, dimG, dimH  = dims

	dimX = dimX1*dimX2

	Pg_h    	= Normalize(rand(dimG, dimH, 1, 1, 1, 1), (idxH, ))

	Ph      	= ph(Pg_h, Pg)

	Ph_g    	= ph_g(Pg_h, Pg, Ph) 

	Phxi_ai 	= phxi_ai(Pgxi_ai, Ph_g)

	Ph_ai   	= sum(Pg_ai .* Ph_g, dims=idxG)

	Phxiai 		= Phxi_ai .* Ph .* Pxi

	i1 = []
	i2 = []
	l  = []

	Pg_h_    = []
	Ph_      = []
	Ph_g_    = []
	Phxi_ai_ = []
	Ph_ai_ = []
	Phxiai_  = []
	Pgh_  = []

	idx = -1

	for i in 1:10
		i1 = []
		i2 = []
		l  = []
		Pg_h_    = []
		Ph_      = []
		Ph_g_    = []
		Phxi_ai_ = []
		Ph_ai_ = []
		Phxiai_  = []
		Pgh_  = []
		for j in 1:3
			Pg_h 		= pg_h(Ph, Pxi, Pgxi_ai, Phxi_ai, Pg_ai, Ph_ai, beta) 

			Ph      	= ph(Pg_h, Pg)

			Ph_g    	= ph_g(Pg_h, Pg, Ph) 

			Phxi_ai 	= phxi_ai(Pgxi_ai, Ph_g)

			Ph_ai   	= sum(Pg_ai .* Ph_g, dims=idxG)

			Phxiai 		= Phxi_ai .* Ph .* Pxi

			Pgh 	= pgh(Pg_h, Pg)

			i1_ 	= Igh(Pgh, Ph, Pg) 

			i2_ 	= Ih_xiai(Phxiai, Ph_ai, Phxi_ai) 

			l_       =  i1_ - beta*i2_

			push!(i1, i1_)
			push!(i2, i2_)
			push!(l, l_)

			push!(Pg_h_    , Pg_h)
			push!(Ph_      , Ph)
			push!(Ph_g_    , Ph_g)
			push!(Phxi_ai_ , Phxi_ai)
			push!(Phxiai_ ,  Phxiai)
			push!(Pgh_  ,    Pgh)
		end

		idx = argmin(l)

		println((i, round.((Float64(i1[idx]), Float64(i2[idx]), Float64(l[idx])), digits=15), round(Float64(beta))))

	end

	return Pg_h_[idx], Ph_g_[idx], Ph_[idx], Phxi_ai_[idx], i1[idx], i2[idx], l[idx]

end

function main()

	dimX1, dimX2, dimA = (10, 10, 9)
	dimX = dimX1*dimX2
	dimG = -1
	dimH = -1

	dims = (dimX1, dimX2, dimA, dimG, dimH)

	C    = CreateWalls(dims, "GlobalRef5x5r9")
	display(C); println()
# 	C    = CreateWalls(dims, "B")
	dimG = dimX#length(G)
dimX1 = dimX1 +1
dimX2 = dimX2 +1
	dims = (dimX1, dimX2, dimA, dimG, dimH)
	G    = GenerateGoalSet(dims, "fullX", C)

	dimH = 9

	Pgxi_ai  = Array{Float64}(undef, dimG,    1, dimX, dimA,    1, 1)
	Phxi_ai  = Array{Float64}(undef,    1, dimH, dimX, dimA,    1, 1)
	Pgxiai_xj= Array{Float64}(undef, dimG,    1, dimX, dimA, dimX, 1)
	Pg       = Array{Float64}(undef, dimG,    1,    1,    1,    1, 1)
	Ph       = Array{Float64}(undef,    1, dimH,    1,    1,    1, 1)
	Ph_g     = Array{Float64}(undef, dimG, dimH,    1,    1,    1, 1)
	Pg_h     = Array{Float64}(undef, dimG, dimH,    1,    1,    1, 1)
	Pxi      = Array{Float64}(undef,    1,    1, dimX,    1,    1, 1)

	Vgx, Qgxa, _Pgxi_ai, _Pgxiai_xj = SolveMazeGoalCond(dims, C, G)
	### Pgxiai_xj[:, 1, :, :, :, 1] = _Pgxiai_xj ###
	### Pgxi_ai[  :, 1, :, :, 1, 1] = _Pgxi_ai 	 ###
	### Pg      = Normalize( ones(dimG, 1, 1, 1, 1, 1), (idxG,) ) ###
	### Pg_BF      = Normalize( ones(BigFloat, dimG, 1, 1, 1, 1, 1), (idxG,) ) ###

	display(size(Vgx)); println()
	display(reshape(Vgx[1, :, 1, 1, 1, 1], (dimX1, dimX2))); println()

	exit(1)

	beta = 500.0
# 	Pgxi_ai, Pxi_ai, Pxi          = pgxi_ai_goal( dims, Pg, Qgxa, Pgxiai_xj, beta )



	@time Pgxi_ai_BF, Pxi_ai_BF, Pxi_BF = pgxi_ai_goal( dims, 	convert(Array{BigFloat}, Pg), 
									convert(Array{BigFloat}, Qgxa), 
									convert(Array{BigFloat}, Pgxiai_xj), 
									convert(BigFloat, beta) )

# 	@time Pgxi_ai_CUBF, Pxi_ai_CUBF, Pxi_CUBF = pgxi_ai_goal( dims, 	convert(CuArray, convert(Array{BigFloat}, Pg)), 
# 							    			convert(CuArray, convert(Array{BigFloat}, Qgxa)), 
# 										convert(CuArray, convert(Array{BigFloat}, Pgxiai_xj)), 
# 										convert(BigFloat, beta) )

	
# 	@time Pgxi_ai_CUBF, Pxi_ai_CUBF, Pxi_CUBF = pgxi_ai_goal( dims, 	cu(convert(Array{BigFloat}, Pg)), 
# 							    			cu(convert(Array{BigFloat}, Qgxa)), 
# 										cu(convert(Array{BigFloat}, Pgxiai_xj)), 
# 										cu(convert(BigFloat, beta)) )
# 	display(reshape(Pxi,    (dimX1, dimX2))); println()

# 	fig, axes = subplots(1, 2)
# 	println(size(axes))
# 	img = axes[1].imshow(reshape(convert(Array{Float64}, Pxi_BF), (dimX1, dimX2)))
# 	fig.colorbar(img, axes[1])
# 
# 	img = axes[2].imshow(reshape(convert(Array{Float64}, Pxi_BF), (dimX1, dimX2)))
# 	fig.colorbar(img, axes[2])
# 
# 	show()
# 	close()
# 
# 
# 	
# 
# 
# 	exit(1)


	# 	display(reshape(Vgx[1, :, :, :, :, :], (dimX1, dimX2))); println()
	# 	display(reshape(Qgxa[1, 1, :, :, 1, 1], (dimX, dimA))); println()
	# 	display(reshape(Pxi[:, :, :, :, :, :], (dimX1, dimX2))); println()
	# 	println((size(Pgxi_ai), size(Pxi)))

	Pg_ai_BF = sum( Pgxi_ai_BF .* Pxi_BF, dims=idxXi )
	# 	Pg_ai = pg_ai( Pgxi_ai, Pxi )


	I1 = []
	I2 = []
	b = -1
	for b in (1.0, 10.0, 100.0, 200.0, 500)	

		Pg_h, Ph_g, Ph, Phxi_ai, i1, i2, l = goalIB(dims, Pg_BF, Pgxi_ai_BF, Pg_ai_BF, Pxi_BF, b)

		push!(I1, i1)
		push!(I2, i2)

		for hi in 1:dimH

			fig = figure()
			img = imshow(reshape(convert(Array{Float64}, Ph_g[:, hi]), (dimX1, dimX2)))
			colorbar(img)
			savefig("./plots/Ph$(hi)_g_x1$(dimX1)x2$(dimX2)h$(dimH)g$(dimG)a$(dimA)b$(b).png", bbox_inches="tight")	
			close(fig)

		end
	end
	
	fig = figure()
	plot(I1, I2)
	savefig("./plots/GIB_x1$(dimX1)x2$(dimX2)h$(dimH)g$(dimG)a$(dimA)b$(b).png", bbox_inches="tight")	
	close(fig)

end

main()

end

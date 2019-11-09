module DP

using LinearAlgebra
using RandomPolicy_tmp
using ProbUtils
using GridWorld
using InfoUtils
using PlotUtils
using PyPlot
using Distributions

export SolveMazeGoalCond

abstract type TSR <: AbstractArray{Float64,any} end

# const gamma = 0.975
const gamma = 1.0
const nDP   = 100

function SolveMazeGoalCond(dims::Tuple, C::Array{Int}, G::Array{Int})

	dimX1, dimX2, dimA = dims 
	dimX = dimX1*dimX2
	dimG = dimX

	actionProb = 1.0

	Pgxi_ai   = Array{Float64}(undef, dimX, dimX, dimA, 1, 1)
	Pgxiai_xj, Rgxiaixj = InitPxiaig_xj(dims, G, C, actionProb)
	Pxiai_xj = Array{Float64}(undef, dimX, dimA, dimX, 1)
	Rxiaixj  = Array{Float64}(undef, dimX, dimA, dimX, 1)
	Vgx      = Array{Float64}(undef, dimX, dimX,    1, 1)

	for g in 1:length(G)
		Pxiai_xj = Pgxiai_xj[g, :, :, :, :]
		Rxiaixj  =  Rgxiaixj[g, :, :, :, :]
		V, Pxi_ai =  SolveMaze(dims, C, Pxiai_xj, Rxiaixj)
		Pgxi_ai[g, :, :, 1, 1] = Pxi_ai
		Vgx[g, :, 1, 1] = V
	end

	return Vgx, Pgxi_ai, Pgxiai_xj
end


function SolveMazeGoalCond(dimX1, dimX2, dimA)

	dimX  = dimX1*dimX2
	dimG  = dimX
	dimH  = dimX
	dims  = (dimX1, dimX2, dimA, dimG, dimH)

	C    = CreateWalls(dims, "B")
	G    = GenerateGoalSet(dims, "fullX", C)

	actionProb = 1.0

	PgxCaC_xN, RgxCaChC_xN = InitPxiaig_xj(dims, G, C, actionProb)
	QghC_hN                = InitAtBetaInfinityQghC_hN(dims)
	Ph_g                   = zeros(Float64, dimG, 1, 1, dimH, 1, 1, 1)
	Ph_g[:, 1, 1, :, 1, 1, 1] = Matrix{Float64}(I, dimG, dimH)

	QgxChC_hN              = InitAtBetaInfinityQgxChC_hN(dims)
	Pxh_g                  = InitPxh_gAtBetaInfinity(dims)

	QxChC_hN               = EncoderMarginal(QgxChC_hN, Pxh_g)
	IghC_hN                = zeros(Float64, (dimG, 1, 1, dimH, 1, 1, dimH))

	TotRgxCaChC_xN = RgxCaChC_xN
	Vgxh, PxChP_aC = BellmanStateIndependent(dims, PgxCaC_xN, TotRgxCaChC_xN, QghC_hN, Ph_g)

	return Vgxh, PxChP_aC 
end

function SolveMaze(dims, C, Pxiai_xj, Rxiaixj)

	dimX1, dimX2, dimA = dims
	dimX = dimX1*dimX2

	actionProb = 1.0
	Xf = dimX
	Xo = 1

	Pxi_ai = Normalize(ones(dimX, dimA, 1, 1), (2, ))	
	Rxiai = Expectation(Rxiaixj, Pxiai_xj, [3])
	Qi     = zeros(dimX, dimA, 1, 1)

	for i in 1:100
		Pxiai_xjaj = broadcast(*, Pxiai_xj, permutedims(Pxi_ai, (3, 4, 1, 2))) 
		Qj = Expectation(Pxiai_xjaj, permutedims(Qi, (3, 4, 1, 2)), [3, 4])
		Qi = Rxiai + Qj 
		if i % 20 == 0
			Pxi_ai = zeros(dimX, dimA, 1, 1)
			# randn: to prevent bias in Q, caused by the order of the actions 
			Pxi_ai[findmax(Qi .+ 0.001*randn(dimX, dimA, 1, 1), dims=2)[2]] .= 1.0 
		end
	end

	return Expectation(Qi, Pxi_ai, [2]), Pxi_ai

end


function Horizon10(dims, Pxiai_xj, Pxi_ai, P)
	dimX1, dimX2, dimA = dims
	dimX = dimX1*dimX2

	P = zeros(dimX, dimA, dimX, dimA, dimX, dimA, dimX, dimA, dimX, dimA)

	for t = 1:10
			
	end

end

function mainSolveMaze()
	dimX1 = 3
	dimX2 = 3
	dimA  = 5

	dimX = dimX1*dimX2
	dims = (dimX1, dimX2, dimA, -1, -1)

	Xf = dimX
	Xo = 1
	actionProb = 1.0
	C    = CreateWalls(dims, "B")
	Pxiai_xj, Rxiaixj =  SetPxj_xiai(dims, actionProb, C, Xf, Xo)

	V, Pxi_ai =  SolveMaze(dims, C, Pxiai_xj, Rxiaixj)
# 	display(reshape(V[:, 1, 1, 1], dimX1, dimX2)); println()
# 	display(Pxi_ai); println();

# 	Pxi_ai   = Normalize(ones(dimX, dimA, 1, 1), (2,))
	Pxj_aj   = permutedims(Pxi_ai, (3, 4, 1, 2))

	Px0      = Normalize(ones(dimX, 1, 1, 1), (1,)) 
# 	Px0      = zeros(dimX, 1, 1, 1); Px0[1] = 1.0;
	Px0_a0x1 = broadcast(*, Pxiai_xj, Pxi_ai)
	Px0_x1   = sum(Px0_a0x1, dims=2) 
	Px0x1    = broadcast(*, Px0, Px0_x1)	
	Px1      = sum(Px0x1, dims=1)
	Px0a0x1  = broadcast(*, Px0_a0x1, Px0)
	Px0a0x1a1= broadcast(*, Px0a0x1,    Pxj_aj)
	Px0a0_x1a1 = broadcast(*, Pxiai_xj, Pxj_aj)
	Px1a1    = sum(Px0a0x1a1, dims=[1, 2])

# 	display(reshape(Px0[:, 1, 1, 1], (dimX1, dimX1))); println()	
# 	display(reshape(Px1[1, 1, :, 1], (dimX1, dimX1))); println()	
# 	display( Px0x1[:, 1, :, 1]); println()
# 	display(Px0_x1[:, 1, :, 1]); println()
	
	Hx1      = -sum(log2.(Px1 .^ Px1))
	Hx0_x1   = -sum(log2.(Px0_x1 .^ Px0x1))
	Hx0a0_x1 = -sum(log2.(Pxiai_xj .^ Px0a0x1 ))
	
	Hx1a1   = -sum(log2.( Px1a1 .^ Px1a1))
	Hx0a0_x1a1   = -sum(log2.( Px0a0_x1a1 .^ Px0a0x1a1))
		
	

	println((Hx1, Hx0_x1, Hx0a0_x1))
	println("I[x1;x0], I[x1;a0|x0], I[x1, a1; x0, a0])=",(Hx1 - Hx0_x1, Hx0_x1 - Hx0a0_x1, Hx1a1 - Hx0a0_x1a1))

end

function mainSolveMazeGoalCond()

	dimX1 = 3
	dimX2 = 3
	dimA  = 5
	dimX = dimX1*dimX2

	dims = (dimX1, dimX2, dimA, dimX, dimX)

	C    = CreateWalls(dims, "B")
	G    = GenerateGoalSet(dims, "fullX", C)

	actionProb = 1.0

	Pgxi_ai   = Array{Float64}(undef, dimX, dimX, dimA, 1, 1)
	Pgxiai_xj, Rgxiaixj = InitPxiaig_xj(dims, G, C, actionProb)
	Pxiai_xj = Array{Float64}(undef, dimX, dimA, dimX, 1)
	Rxiaixj  = Array{Float64}(undef, dimX, dimA, dimX, 1)

	for g in 1:length(G)
		Pxiai_xj = Pgxiai_xj[g, :, :, :, :]
		Rxiaixj  =  Rgxiaixj[g, :, :, :, :]

		V, Pxi_ai =  SolveMaze(dims, C, Pxiai_xj, Rxiaixj)
		Pgxi_ai[g, :, :, 1, 1] = Pxi_ai
# 		display(reshape(V[:, 1, 1, 1], dimX1, dimX2)); println()
	end
	# 	display(Pxi_ai); println();

# 	Pgxi_ai   = Normalize(ones(dimX, dimX, dimA, 1, 1), (3,))
# 	println(size( Pgxi_ai ))
	Pgxj_aj   = permutedims(Pgxi_ai, (1, 4, 5, 2, 3))

	Px0      = Normalize(ones(1, dimX, 1, 1, 1), (2,)) 
	Pg       = Normalize(ones(dimX, 1, 1, 1, 1), (1,)) 
	Pgx0     = broadcast(*, Px0, Pg)

	# 	Px0      = zeros(dimX, 1, 1, 1); Px0[1] = 1.0;

	Pgx0_a0x1   = broadcast(*, Pgxiai_xj, Pgxi_ai)
	Pgx0a0_x1a1 = broadcast(*, Pgxiai_xj, Pgxj_aj) 

	Px0_a0x1      = sum(broadcast(*, Pgx0_a0x1, Pg), dims=1)
	Px0a0_x1a1    = sum(broadcast(*, Pgx0a0_x1a1, Pg), dims=1)

	Px0_x1   = sum(Px0_a0x1, dims=[3]) 
	Px0x1    = broadcast(*, Px0, Px0_x1)	
	Px1      = sum(Px0x1, dims=4)

	Px0a0x1  = broadcast(*, Px0_a0x1, Px0)
	Px0a0x1a1= broadcast(*, Px0a0x1,    Pgxj_aj)
# 	Px0a0_x1a1 = broadcast(*, Pxiai_xj, Pgxj_aj)
	Px1a1    = sum(Px0a0x1a1, dims=[1, 2])

	Hx1      = -sum(log2.(Px1 .^ Px1))
	Hx0_x1   = -sum(log2.(Px0_x1 .^ Px0x1))

# 	println((Hx1, Hx0_x1))

# 	Hx0a0_x1 = -sum(log2.(Pxiai_xj .^ Px0a0x1 ))
	Hx1a1   = -sum(log2.( Px1a1 .^ Px1a1))
	println((size(Px0a0_x1a1), size(Px0a0x1a1)))
	Hx0a0_x1a1   = -sum(log2.( Px0a0_x1a1 .^ Px0a0x1a1))
# 		
# 	
# 
# 	println((Hx1, Hx0_x1, Hx0a0_x1))
	println("I[x1;x0], I[x1, a1; x0, a0])=", (Hx1 - Hx0_x1, Hx1a1 - Hx0a0_x1a1))


end

function BellmanIB(	dims::Tuple,
	Pgxiai_xj::Array{Float64},
	R::Array{Float64},
	Qgxa_h::Array{Float64},
	Pxa_g::Array{Float64}, Px_g::Array{Float64}, beta::Float64, gamma::Float64)


	dimX1, dimX2, dimA, dimG, dimH = dims
	dimX = dimX1*dimX2

	Lcurn    = zeros(dimG, dimX,  dimA, dimH, 1, 1, 1)
	Lnext    = []
	Lxah     = []
	LnextExpected = []
	Pxih_ai = InitProb((1, dimX, dimA, dimH, 1, 1, 1), (3,))

	Qxa_hg = JointProb(Qgxa_h, Pxa_g)


	# 	display(Qxa_hg)
	# 	println()

	if ~isempty(findall(iszero, Qxa_hg))
		println("~isempty(findall(izsero, Qxa_hg))")
		exit(1)
	end

	Qxa_h  = Marginal(Qxa_hg, (1,))
	Qxah_g = broadcast(/, Qxa_hg, Qxa_h)

	Iq = log.(broadcast(/, Qgxa_h, Qxa_h))		

	for gi = 1:dimG, xi = 1:dimX
		if xi == gi
			Iq[gi, xi, :, :, 1, 1, 1] .= 0.0
		end
	end

	# 		println("Iq, $(maximum(Iq)), $(minimum(Iq)), $(mean(Iq))")

	r  = broadcast(+, (1/beta)*Iq, -(1)*R)

	for iter = 1:500
		Pxjh_aj = permutedims(Pxih_ai, (1, 5, 6, 4, 2, 3, 7))
		Lnext   = permutedims(Lcurn,   (1, 5, 6, 4, 2, 3, 7))
		Pnext   = JointProb(Pgxiai_xj, Pxjh_aj)
		LnextExpected = Expectation(Pnext, Lnext, [5, 6])
		Lcurn   = -(r + gamma*LnextExpected)

		if mod(iter, 250) == 0
			Pxih_ai .= 0.0
			Lxah = Expectation(Lcurn, Qxah_g, [1])
			# 			Lxah = Expectation(Lcurn, Px_g, [1])
			# 			Lxah = Expectation(Lcurn, Pxa_g, [1])
			Pxih_ai[findmax(Lxah, dims=3)[2]] .= 1.0 
		end
		# 		println("$(iter), $(maximum(Lcurn)), $(minimum(Lcurn)), $(mean(Lcurn))")
	end
	# exit(1)
	# 	fi = figure()
	# fig = PyPlot.imshow(reshape(Lcurn[1, 1, 1, :], (dimX1, dimX2)), interpolation="none", extent=[1, dimX1, 1, dimX2])
	#  colorbar(fig, ticks=[minimum(Lcurn[1, 1, 1, :]), maximum(Lcurn[1, 1, 1, :])])
	# 	show()
	# 	close(fi)

	return Lcurn, Pxih_ai, LnextExpected, Lxah

end

function BellmanEntropyEncoderStateAction(	dims::Tuple, 
	PgxCaC_xN::Array{Float64},
	R::Array{Float64},  
	QgxChC_hN::Array{Float64}, 
	Pg::Array{Float64})

	println("start BellmanStateDependent at $(@__MODULE__), $(@__LINE__)")

	dimX1, dimX2, dimA, dimG, dimH = dims
	dimX = dimX1*dimX2

	Vcurn    = zeros(Float64, dimG, dimX,    1, dimH, 1, 1, 1)
	PxChP_aC = InitProb((1, dimX, dimA, dimH, 1, 1, 1), (3,))

	for iter in 1:nDP
		Vnext  = permutedims(Vcurn, (1, 5, 3, 7, 2, 6, 4))
		Rnext  = broadcast(+, R, gamma*Vnext)
		Ptrans = JointProb(PgxCaC_xN, QgxChC_hN)
		Qgxah  = Expectation(Rnext, Ptrans, [5, 7])
		Vcurn  = Expectation(Qgxah, PxChP_aC, [3]) # Vgxh

		if mod(iter, 10) == 0
			# 			println("[$(@__MODULE__), $(@__LINE__)]: iter[$(iter)]")

			# 			Pxh_g = Normalize(exp.(-Vcurn), (1,))
			PxChP_aC .= 0.0
			Qxah = Expectation(Qgxah, Pxh_g, [1])
			PxChP_aC[findmax(Qxah, dims=3)[2]] .= 1.0 
		end
	end

	return Vcurn, PxChP_aC, Pxh_g
end


function LogValueIterationGivenEncoder( dims::Tuple, 
	PgxCaC_xN::Array{Float64},
	R::Array{Float64},  
	Pg::Array{Float64}, 
	Qgxiai_h::Array{Float64},
	gamma::Float64, beta::Float64 )

	dimX1, dimX2, dimA, dimG, dimH = dims
	dimX = dimX1*dimX2

	Qgxiai_h, Z     = EntropyEncoder(dims, Pxih_ai, PgxCaC_xN, Lxiai, gamma)		

	Hgsa_h = -sum(Qgxiai_h .* log.(Qgxiai_h), dims=7)

	r       = Expectation(Hgsa_h + beta*Expectation(PgxCaC_xN, R, [5]), [1])


	Lxiai 		= zeros(1, dimX, dimA, 1, 1, 1, 1 )			
	Lxiai_prev      = Lxiai
	Pxjh_aj         = RandomPolicy(dims, true) 

	for i = 1:100
		Pxiai_xjai = sum(JointProb(Pg,JointProb(Qgxiai_h,JointProb(PgxCaC_xN, Pxjh_aj))),dims=[1, 7])
		TEST_PROB_UNITY(Pxiai_xjai, "Pxiai_xjai", "LogValueIterationGivenEncoder", [5, 6])
		exit(1)

	end




end

function LogValueIterationTest(	dims::Tuple, 
	PgxCaC_xN::Array{Float64},
	R::Array{Float64},  
	Pg::Array{Float64}, 
	gamma::Float64, beta::Float64 )

	dimX1, dimX2, dimA, dimG, dimH = dims
	dimX = dimX1*dimX2
	aStr = ActionStrings(dimA)
	r               = Expectation(PgxCaC_xN, R, [5])
	beta = 10.00000010
	gamma = 0.97
	Lgxiai 		= zeros(dimG, dimX, dimA, 1, 1, 1, 1 )			
	# 	Lgxiai_prev 	= Lgxiai
	# 	Pxih_ai         = Normalize(ones(1, dimX, dimA, dimH, 1, 1, 1), (3,))
	hgxMAX = 100.0
	hgx = 0.0
	vgx = 0.0
	Vgx = []
	Hgx = []
	Q = []
	Pxih_ai = []
	Pxjh_aj = []
	Harr = []
	Varr = []
	Lgxjaj = []
	for beta = 2:0.001:15
		# 	for beta = 2:0.05:15
		# 	Lgxiai 		= zeros(dimG, dimX, dimA, 1, 1, 1, 1 )			
		for iSample = 1:1

			Pxih_ai = Normalize(ones(1, dimX, dimA, dimH, 1, 1, 1), (3,))

			for i = 1:100

				Lgxjaj  = permutedims(Lgxiai, (1, 5, 6, 4, 2, 3, 7))	
				Pxjh_aj = permutedims(Pxih_ai, (1, 5, 6, 4, 2, 3, 7)) 
				expArgInner  = Expectation(Pxjh_aj, Lgxjaj, [6])

				expArg  = Expectation(PgxCaC_xN, expArgInner, [5])

				Lgxiai  = (-log.(sum(exp.(-gamma*expArg), dims=[4])) - beta*r)

				if mod(i, 50) == 0
					# if false
					Pxih_ai = zeros(dimG, dimX, dimA, 1, 1, 1, 1)
					Pxih_ai[findmax(Lgxiai, dims=3)[2]] .= 1.0 
					Pxih_ai = Normalize(Pxih_ai .+ 0.1, (3,))
					Vgx = Expectation(Lgxiai, Pxih_ai, [3]) 
					vgx = sum(Vgx)/(dimX*dimG)
					Pxih_ai = permutedims(Pxih_ai, (4, 2, 3, 1, 5, 6, 7))
					expArgInner  = Expectation(Pxjh_aj, Lgxjaj, [6])
					expArg  = Expectation(PgxCaC_xN, expArgInner, [5])
					Q = broadcast(/, exp.(-gamma*expArg), sum(exp.(-gamma*expArg), dims=[4]))
					H = -sum(log.(Q.^Q) , dims=[4])
					tmp = permutedims(Pxih_ai, (4, 2, 3, 1, 5, 6, 7))
					Hgx = Expectation(H, tmp, [4])
					hgx = sum(Hgx)/(dimX*dimG)
				end

			end
			# 			if hgx > hgxMAX
			# 				tmp11  = hgx
			# 				hgx = hgxMAX 
			# 				hgxMAX = tmp11
			# 			end
			# 				if mod(i, 10) == 0
			if false
				Pxih_ai = zeros(dimG, dimX, dimA, 1, 1, 1, 1)
				Pxih_ai[findmax(Lgxiai, dims=3)[2]] .= 1.0 
				Pxih_ai = Normalize(Pxih_ai .+ 0.1, (3,))
				Vgx = Expectation(Lgxiai, Pxih_ai, [3]) 
				vgx = sum(Vgx)/(dimX*dimG)
				Pxih_ai = permutedims(Pxih_ai, (4, 2, 3, 1, 5, 6, 7))

				expArgInner  = Expectation(Pxjh_aj, Lgxjaj, [6])
				expArg  = Expectation(PgxCaC_xN, expArgInner, [5])
				Q = broadcast(/, exp.(-gamma*expArg), sum(exp.(-gamma*expArg), dims=[4]))
				H = -sum(log.(Q.^Q) , dims=[4])
				tmp = permutedims(Pxih_ai, (4, 2, 3, 1, 5, 6, 7))
				Hgx = Expectation(H, tmp, [4])

				hgx = sum(Hgx)/(dimX*dimG)
			end
			if false

				display(Lgxiai[7, :, :, 1, 1, 1, 1])
				println()

				Pxih_ai = zeros(dimG, dimX, dimA, 1, 1, 1, 1)
				Pxih_ai[findmax(Lgxiai, dims=3)[2]] .= 1.0 
				Vgx = Expectation(Lgxiai, Pxih_ai, [3]) 
				Pxih_ai = permutedims(Pxih_ai, (4, 2, 3, 1, 5, 6, 7))


				for xi = 1:dimX
					println(aStr[findall(!iszero, Pxih_ai[1, xi, :, 7, 1, 1, 1])])
				end
			end

		end
		if true

			display(Lgxiai[7, :, :, 1, 1, 1, 1])
			println()

			Pxih_ai = zeros(dimG, dimX, dimA, 1, 1, 1, 1)
			Pxih_ai[findmin(Lgxiai, dims=3)[2]] .= 1.0 
			Vgx = Expectation(Lgxiai, Pxih_ai, [3]) 
			Pxih_ai = permutedims(Pxih_ai, (4, 2, 3, 1, 5, 6, 7))


			for xi = 1:dimX
				println(aStr[findall(!iszero, Pxih_ai[1, xi, :, 7, 1, 1, 1])])
			end
		end
		# 	H = -sum(log.(Q.^Q) , dims=[4])
		# 	tmp = permutedims(Pxih_ai, (4, 2, 3, 1, 5, 6, 7))
		# 	Hgx = Expectation(H, tmp, [4])

		# 	Vgx = Expectation(Lgxiai, Pxih_ai, [3]) 

		if ~isempty(findall( Vgx .> 0.0))
			break
		end

		display(reshape(Vgx[7, :, 1, 1, 1, 1, 1] .- 0*maximum(Vgx[7, :, 1, 1, 1, 1, 1]), (dimX1, dimX2)))
		println()
		display(reshape(Hgx[7, :, 1, 1, 1, 1, 1] .- 0*maximum(Hgx[7, :, 1, 1, 1, 1, 1]), (dimX1, dimX2)))
		println()

		display((beta, sum(Hgx)/(dimX*dimG), sum(Vgx)/(dimX*dimG)))
		println()

		push!(Harr, hgx)
		push!(Varr, vgx)
	end

	Varr = Varr 

	# PyPlot.plot(reverse(Harr), reverse(Varr))
	fh = figure(1)
	PyPlot.plot(1.44*(Harr[2:end-1]), abs.(Varr[2:end-1]), label=string(L"tempreture, T=1/\beta,", " increases from left to right"))
	# 	PyPlot.ylabel(string(L"Q(s, a)", ", average number of steps to the goal"))
	PyPlot.ylabel(string("average number of steps to the goal"))
	# 	PyPlot.xlabel(L"{H}(h\mid g, s, a) = I[h, g\mid s, a] - {H}_{constant\;prior}(h\mid s, a)")
	# 	PyPlot.xlabel(string(L"{H}(h\mid g, s, a)", ", uncertanty (in bits) of ", L"h", " given ", L"g, s, a"))
	PyPlot.xlabel(string("uncertanty (in bits) of ", L"h", " given ", L"g, s, a"))
	plt.legend()
	# 	plt.title(L"L(s, a)=I[h, g\mid s, a] - \beta Q(s, a)")
	plt.title(string(L"g\in G, h\in H", " : original, compressed goals"))
	# 	PyPlot.plot((Varr[2:end-1]), (Harr[2:end-1]))
	savefig(string("./plotsDP/", "HvsV.png"), bbox_inches="tight")
	close(fh)
	# show()


	return Lgxiai 

end
function LogValueIteration(	dims::Tuple, 
	PgxCaC_xN::Array{Float64},
	R::Array{Float64},  
	Pg::Array{Float64}, 
	gamma::Float64, beta::Float64 )

	dimX1, dimX2, dimA, dimG, dimH = dims
	dimX = dimX1*dimX2

	r               = Expectation(PgxCaC_xN, R, [5])

	Lxiai 		= zeros(1, dimX, dimA, 1, 1, 1, 1 )			
	Lxiai_prev 		= Lxiai
	Pxih_ai         = RandomPolicy(dims) 

	for i = 1:100
		_, Z 		=  EntropyEncoder(dims, Pxih_ai, PgxCaC_xN, Lxiai, gamma)
		Lxiai 		= -Expectation(Pg, log.(Z) +  beta*r, [1]) 	
		display(Lxiai[1, :, :, 1, 1, 1, 1])
		# 		if i > 1
		println(abs(maximum(Lxiai - Lxiai_prev)))
		# 		end
		Lxiai_prev = Lxiai
	end

	return Lxiai 

end

function BellmanStateDependent(	dims::Tuple, 
	PgxCaC_xN::Array{Float64},
	R::Array{Float64},  
	QgxChC_hN::Array{Float64}, Pxh_g::Array{Float64})

	println("start BellmanStateDependent at $(@__MODULE__), $(@__LINE__)")

	dimX1, dimX2, dimA, dimG, dimH = dims
	dimX = dimX1*dimX2

	Vcurn    = zeros(Float64, dimG, dimX,    1, dimH, 1, 1, 1)
	PxChP_aC = InitProb((1, dimX, dimA, dimH, 1, 1, 1), (3,))

	Qxah = zeros(Float64, 1, dimX,  dimA,  1, dimH, 1, 1, 1)

	for iter in 1:nDP
		Vnext  = permutedims(Vcurn, (1, 5, 3, 7, 2, 6, 4))
		Rnext  = broadcast(+, R, gamma*Vnext)
		Ptrans = JointProb(PgxCaC_xN, QgxChC_hN)
		Qgxah  = Expectation(Rnext, Ptrans, [5, 7])
		Vcurn  = Expectation(Qgxah, PxChP_aC, [3]) # Vgxh

		if mod(iter, 10) == 0
			PxChP_aC .= 0.0
			Qxah = Expectation(Qgxah, Pxh_g, [1])
			if true #last working
				PxChP_aC[findmax(Qxah, dims=3)[2]] .= 1.0 
			else
				# 				maxIds =  0.0
				# 				findmax(A)[2]
			end
		end
	end

	return Vcurn, PxChP_aC, Pxh_g, Qxah
end

function BellmanStateIndependent(	dims::Tuple, 
	PgxCaC_xN::Array{Float64},
	R::Array{Float64},  
	QghC_hN::Array{Float64}, Ph_g::Array{Float64})

	println("start BellmanStateIndependent at $(@__MODULE__), $(@__LINE__)")

	dimX1, dimX2, dimA, dimG, dimH = dims
	dimX = dimX1*dimX2

	Vcurn    = zeros(Float64, dimG, dimX,    1, dimH, 1, 1, 1)
	PxChP_aC = InitProb((1, dimX, dimA, dimH, 1, 1, 1), (3,))

	for iter in 1:nDP
		Vnext  = permutedims(Vcurn, (1, 5, 3, 7, 2, 6, 4))
		Rnext  = broadcast(+, R, gamma*Vnext)
		Ptrans = JointProb(PgxCaC_xN, QghC_hN)
		Qgxah  = Expectation(Rnext, Ptrans, [5, 7])
		Vcurn  = Expectation(Qgxah, PxChP_aC, [3]) # Vgxh

		if mod(iter, 10) == 0
			println("[$(@__MODULE__), $(@__LINE__)]: iter[$(iter)]")
			PxChP_aC .= 0.0
			Qxah = Expectation(Qgxah, Ph_g, [1])
			PxChP_aC[findmax(Qxah, dims=3)[2]] .= 1.0 
		end
	end

	return Vcurn, PxChP_aC
end



function main()

	dimX1 = 3
	dimX2 = 3
	dimX = dimX1*dimX2
	dimA  = 5
	dimG  = dimX
	dimH  = dimX
	dims = (dimX1, dimX2, dimA, dimG, dimH)

	C    = CreateWalls(dims, "B")
	G    = GenerateGoalSet(dims, "fullX", C)

	actionProb = 1/dimA
	# 	actionProb = 1.0
	actionProb = 0.8

	PgxCaC_xN, RgxCaChC_xN = InitPxiaig_xj(dims, G, C, actionProb)
	QghC_hN                = InitAtBetaInfinityQghC_hN(dims)
	Ph_g                   = zeros(Float64, dimG, 1, 1, dimH, 1, 1, 1)
	Ph_g[:, 1, 1, :, 1, 1, 1] = Matrix{Float64}(I, dimG, dimH)

	QgxChC_hN              = InitAtBetaInfinityQgxChC_hN(dims)
	Pxh_g                  = InitPxh_gAtBetaInfinity(dims)

	Pxh_g = Normalize(Pxh_g .+ 0.01, (1,))

	# 	QxChC_hN               = zeros(1, dimX, 1, dimH, 1, 1, dimH)
	# 	for xi in 1:dimX
	# 		QxChC_hN[1, xi, 1, :, 1, 1, :] = ones(dimH, dimH)/dimH
	# 	end

	# 	Pxh_g = zeros(dimG, dimX, 1, dimH, 1, 1, 1)
	# 	for xi in 1:dimX
	# 		Pxh_g[:, xi, 1, :, 1, 1, 1] = ones(dimG, dimH)/dimG
	# 	end

	QxChC_hN               = EncoderMarginal(QgxChC_hN, Pxh_g)


	# 	println(QxChC_hN[1, 1, 1, 1:3, 1, 1, :])
	# 	exit(1)

	# 	IgxCaChC               = zeros(Float64, (dimG, dimX, dimA, 1, 1, 1, dimH))
	# 	IghC_hN                = ones(Float64, (dimG, 1, 1, dimH, 1, 1, dimH))
	IghC_hN                = zeros(Float64, (dimG, 1, 1, dimH, 1, 1, dimH))

	beta = 2.0
	TotRgxCaChC_xN         = broadcast(+, beta*RgxCaChC_xN, IghC_hN)

	stateIndependent = false
	if stateIndependent	
		Vgxh, PxChP_aC = BellmanStateIndependent(dims, PgxCaC_xN, TotRgxCaChC_xN, QghC_hN, Ph_g)
		Vxh             = Expectation(Vgxh, Ph_g, [1])
		stepSize = AvrgStepSize(dims, Vxh)	
		printGridVxh(dims, Vxh, 100, "StateIndpnd", 1.0)
		println("\nstepSize=$(stepSize)")
	end
	Vgxh, PxChP_aC, Pxh_g  = BellmanStateDependent(dims, PgxCaC_xN, TotRgxCaChC_xN, QgxChC_hN, Pxh_g)
	Vxh             = Expectation(Vgxh, Pxh_g, [1])
	stepSize = AvrgStepSize(dims, Vxh)	
	# 	printGridVxh(dims, Vxh, 100, "StateDpnd", 1.0)
	println("\nstepSize=$(stepSize)")
	# exit(1)
	# 	QgxChC_hN = Encoder(dims, QxChC_hN, Vgxh, PgxCaC_xN, PxChP_aC, gamma)

	# 	println(QgxChC_hN[1, 1, 1, 1, 1, 1, :])
	# 	exit(1)

	# 	QxChC_hN  = EncoderMarginal(QgxChC_hN, Pxh_g)

	QgxChC_hN, QxChC_hN = BA(dims, QxChC_hN, Vgxh, PgxCaC_xN, PxChP_aC, Pxh_g, gamma)  

	# 	println("QgxChC_hN=\n", QgxChC_hN)
	# 	println("QxChC_hN=\n", QxChC_hN)

	MI = UpdateMI(dims, QgxChC_hN, Pxh_g )
	# 	println("MI=")
	# 	display(MI)


	for i in 1:50
		Vgxh, PxChP_aC, Pxh_g  = BellmanStateDependent(dims, PgxCaC_xN, TotRgxCaChC_xN, QgxChC_hN, Pxh_g)
		Vxh             = Expectation(Vgxh, Pxh_g, [1])
		stepSize = AvrgStepSize(dims, Vxh)	
		# 	printGridVxh(dims, Vxh, 100, "StateDpnd", 1.0)
		println("\nstepSize=$(stepSize)")

		QgxChC_hN, QxChC_hN = BA(dims, QxChC_hN, Vgxh, PgxCaC_xN, PxChP_aC, Pxh_g, gamma)
	end
end

function mainEntropyEncoder()

	dimX1 = 5
	dimX2 = 5
	dimX  = dimX1*dimX2
	dimA  = 5
	dimG  = dimX
	dimH  = dimX
	dims = (dimX1, dimX2, dimA, dimG, dimH)

	C    = CreateWalls(dims, "B")
	G    = GenerateGoalSet(dims, "fullX", C)

	actionProb = 1/dimA
	actionProb = 1.0
	beta = 1.0

	PgxCaC_xN, RgxCaChC_xN = InitPxiaig_xj(dims, G, C, actionProb)
	QghC_hN                = InitAtBetaInfinityQghC_hN(dims)
	Ph_g                   = zeros(Float64, dimG, 1, 1, dimH, 1, 1, 1)
	Ph_g[:, 1, 1, :, 1, 1, 1] = Matrix{Float64}(I, dimG, dimH)

	QgxChC_hN              = InitAtBetaInfinityQgxChC_hN(dims)
	Pxh_g                  = InitPxh_gAtBetaInfinity(dims)

	Pg   			= Normalize(ones(dimG, 1, 1, 1, 1, 1, 1), (1,))

	IghC_hN                = zeros(Float64, (dimG, 1, 1, dimH, 1, 1, dimH))

	# 	beta = 2.0
	# 	TotRgxCaChC_xN         = broadcast(+, beta*RgxCaChC_xN, IghC_hN)
	TotRgxCaChC_xN         = beta*RgxCaChC_xN

	Vgxh, PxChP_aC, Pxh_g, Qxah = BellmanStateDependent(dims, PgxCaC_xN, TotRgxCaChC_xN, QgxChC_hN, Pxh_g)
	Vxh             	    = Expectation(Vgxh, Pxh_g, [1])

	Qgxa = permutedims(Qxah, (4, 2, 3, 1, 5, 6, 7))
	Pgxi_ai = permutedims(PxChP_aC, (4, 2, 3, 1, 5, 6, 7))
	Vgx = Expectation(Qgxa, Pgxi_ai, [3])
	Qxa = Expectation(Qgxa, Pg, [1])

	for hi = 1:dimX
		display( reshape( Vxh[1, :, 1, hi, 1, 1, 1], (dimX1, dimX2) ) )
		println("\nVgx:")
		display( reshape( Vgx[hi, :, 1, 1, 1, 1, 1], (dimX1, dimX2) ) )
		println()
	end

	println("\nVx:")
	Vx = Expectation(Vgx, Pg, [1])
	# 	Vx = sum(Vgx, dims=1)
	display( reshape( Vx[1, :, 1, 1, 1, 1, 1], (dimX1, dimX2) ) )
	println()
	display(Vgx)
	println()
	display(sum(Vgx, dims=1))
	plotGridFxiai(dims, Qxa, 897, "SSS", beta)

	exit(1)
	# 	Lxiai = LogValueIteration(dims, PgxCaC_xN, RgxCaChC_xN, Pg, gamma, beta)
	# 	plotGridFxiai(dims, Lxiai, 777, "Lxiai", beta)

	# 	LogValueIterationGivenEncoder(dims, PgxCaC_xN, R, Pg, Qgxiai_h, gamma, beta)
end




function mainLogValueIterationTest()

	dimX1 = 3
	dimX2 = 3
	dimX = dimX1*dimX2
	dimA  = 5
	dimG  = dimX
	dimH  = dimX
	dims = (dimX1, dimX2, dimA, dimG, dimH)

	actionProb = 1/dimA
	rctionProb = 1.0
	actionProb = 0.75

	beta = 1.0
	gamma = 0.975
	gamma = 1.0

	C    = CreateWalls(dims, "B")
	G    = GenerateGoalSet(dims, "fullX", C)

	PgxCaC_xN, RgxCaChC_xN = InitPxiaig_xj(dims, G, C, actionProb)
	Pg = Normalize(ones(dimG, 1, 1, 1, 1, 1, 1), (1,))

	Lgxiai = LogValueIterationTest(	dims, PgxCaC_xN, RgxCaChC_xN,  Pg, gamma,beta)
end


function mainIB()

	dimX1 = 3
	dimX2 = 3
	dimX = dimX1*dimX2
	dimA  = 5
	dimG  = dimX
	dimH  = dimX
	dims = (dimX1, dimX2, dimA, dimG, dimH)

	C    = CreateWalls(dims, "B")
	G    = GenerateGoalSet(dims, "fullX", C)

	actionProb = 1/dimA
	actionProb = 1.0
	# 	actionProb = 0.8

	PgxCaC_xN, RgxCaChC_xN = InitPxiaig_xj(dims, G, C, actionProb)
	QghC_hN                = InitAtBetaInfinityQghC_hN(dims)
	Ph_g                   = zeros(Float64, dimG, 1, 1, dimH, 1, 1, 1)
	Ph_g[:, 1, 1, :, 1, 1, 1] = Matrix{Float64}(I, dimG, dimH)

	IghC_hN                = zeros(Float64, (dimG, 1, 1, dimH, 1, 1, dimH))

	beta = 1.0
	TotRgxCaChC_xN         = broadcast(+, beta*RgxCaChC_xN, IghC_hN)

	Vgxh, PxChP_aC  = BellmanStateIndependent(dims, PgxCaC_xN, TotRgxCaChC_xN, QghC_hN, Ph_g)
	Vxh             = Expectation(Vgxh, Ph_g, [1])
	stepSize 	= AvrgStepSize(dims, Vxh)	
	printGridVxh(dims, Vxh, 100, "StateIndpnd", 1.0)
	println("\nstepSize=$(stepSize)")

	PxC_g = Normalize(ones(dimG, dimX, 1, 1, 1, 1, 1), (1,)) 
	PgxC_aC = permutedims(PxChP_aC, [4, 2, 3, 1, 5, 6, 7])
	Px_ag = broadcast(*, PgxC_aC, PxC_g)
	# 	for xi = 1:dimX
	# 	display(Px_ag[:, xi, :, 1, 1, 1, 1])
	# 	println()
	# end

	Pxg_h, Px_h, Pxh_a, Pxh_g = IB(dims, Px_ag, 100.0)

	display(PxC_g[:, :, 1, 1, 1, 1, 1]')
	println()
	display(Px_h[1, :, 1, :, 1, 1, 1])

	A1 = permutedims(PxC_g[:, :, 1, 1, 1, 1, 1], (2, 1))
	A2 = Px_h[1, :, 1, :, 1, 1, 1]
	plotPxi_xj(dims, A1, 333, "Px_g", 1.0)
	plotPxi_xj(dims, A2, 333, "Px_h", 1.0)

	plotGridPx_ag(dims, Px_ag, 444, "Px_ag", 1.0)

	Px_ah = broadcast(*, Pxh_a, Px_h)

	A3 = permutedims(Px_ah, (4, 2, 3, 1, 5, 6, 7))
	plotGridPx_ag(dims, A3, 444, "Px_ah", 1.0)



	# 	Pgxi_xj = sum(broadcast(*, PgxCaC_xN, PxChP_aC), dims=3)
	# 	for hi in dimH
	# 		println("hi=$(hi)")
	# 		display((Pgxi_xj[1, :, 1, hi, :, 1, 1]))
	# 		println()
	# 		display((Pgxi_xj[1, :, 1, hi, :, 1, 1]^8))
	# 		println()
	# 		display((Pgxi_xj[1, :, 1, hi, :, 1, 1]^9))
	# 		println()
	# 		display((Pgxi_xj[1, :, 1, hi, :, 1, 1]^10))
	# 		println()
	# 	end

	exit(1)

end

function mainEstimatePxa_g()

	dimX1 = 3
	dimX2 = 3
	dimX = dimX1*dimX2
	dimA  = 5
	dimG  = dimX
	dimH  = dimX
	dims = (dimX1, dimX2, dimA, dimG, dimH)

	C    = CreateWalls(dims, "B")
	G    = GenerateGoalSet(dims, "fullX", C)

	actionProb = 1.0

	Pgxiai_xj, Rgxiaixj       = InitPxiaig_xj(dims, G, C, actionProb)
	Ph_g                      = zeros(Float64, dimG, 1, 1, dimH, 1, 1, 1)
	Ph_g[:, 1, 1, :, 1, 1, 1] = Matrix{Float64}(I, dimG, dimH)

	QgxChC_hN              = InitAtBetaInfinityQgxChC_hN(dims)
	Pxh_g                  = InitPxh_gAtBetaInfinity(dims)

	IghC_hN                = zeros(Float64, (dimG, 1, 1, dimH, 1, 1, dimH))
	beta = 1.0
	R         = broadcast(+, beta*Rgxiaixj, IghC_hN)

	Vgxh, PxChP_aC, Pxh_g  = BellmanStateDependent(dims, Pgxiai_xj, R, QgxChC_hN, Pxh_g)

	Vxh             = Expectation(Vgxh, Ph_g, [1])
	stepSize = AvrgStepSize(dims, Vxh)	
	printGridVxh(dims, Vxh, 100, "StateIndpnd", 1.0)



	Pgxi_ai  = permutedims(PxChP_aC, (4, 2, 3, 1, 5, 6, 7))
	Pgsa     =  EstimatePxa_g(dims, Pgxiai_xj, Pgxi_ai)


	Pxa_g = broadcast(/, Pgsa, sum(Pgsa, dims=1))
	Psg   = Marginal(Pgsa, (3,))
	Ps_g  = broadcast(/, Psg, sum(Psg, dims=2))

	# println(sum(Pxa_g, dims=1))
	# exit(1)
	aStr = ActionStrings(dimA)
	# println(size(Ps_g))
	# println(size(Pxa_g))
	# 
	# display(Pgsa)
	# exit(1)

	for xi = 1:dimX
		println("P(g | xi=$(xi)):")
		display(reshape(Ps_g[:, xi, 1, 1, 1, 1, 1], (dimX1, dimX2)))
		println()
	end


	for xi = 1:dimX, ai = 1:dimA
		println("xi=$(xi), $(aStr[ai])")
		display(reshape(Pxa_g[:, xi, ai, 1, 1, 1, 1], (dimX1, dimX2)))
		println()
	end
end


function mainBellmanIB()

	dimX1 = 3
	dimX2 = 3
	dimX = dimX1*dimX2
	dimA  = 5
	dimG  = dimX
	dimH  = dimX
	dims = (dimX1, dimX2, dimA, dimG, dimH)

	C    = CreateWalls(dims, "B")
	G    = GenerateGoalSet(dims, "fullX", C)

	actionProb = 0.8

	Pgxiai_xj, Rgxiaixj       = InitPxiaig_xj(dims, G, C, actionProb)
	Ph_g                      = zeros(Float64, dimG, 1, 1, dimH, 1, 1, 1)
	Ph_g[:, 1, 1, :, 1, 1, 1] = Matrix{Float64}(I, dimG, dimH)

	QgxChC_hN              = InitAtBetaInfinityQgxChC_hN(dims)
	Pxh_g                  = InitPxh_gAtBetaInfinity(dims)

	IghC_hN                = zeros(Float64, (dimG, 1, 1, dimH, 1, 1, dimH))
	beta = 1.0
	R         = broadcast(+, beta*Rgxiaixj, IghC_hN)

	Vgxh, PxChP_aC, Pxh_g  = BellmanStateDependent(dims, Pgxiai_xj, R, QgxChC_hN, Pxh_g)

	printBellmanSolution = false
	if printBellmanSolution
		Vxh             = Expectation(Vgxh, Ph_g, [1])
		stepSize = AvrgStepSize(dims, Vxh)	
		printGridVxh(dims, Vxh, 100, "StateIndpnd", 1.0)
	end


	Pgxi_ai  = permutedims(PxChP_aC, (4, 2, 3, 1, 5, 6, 7))
	Pgsa     =  EstimatePxa_g(dims, Pgxiai_xj, Pgxi_ai)

	if length(findall(iszero, Pgsa)) > 0
		exit(1)
	end

	Pxa_g = broadcast(/, Pgsa, sum(Pgsa, dims=1))
	if ~isempty(findall(iszero, Pxa_g))
		println("~isempty(findall(iszero, Pxa_g))")
		exit(1)
	end

	Psg   = Marginal(Pgsa, (3,))
	Ps_g  = broadcast(/, Psg, sum(Psg, dims=2))

	if ~isempty(findall(iszero, Pxa_g))
		display(Pxa_g)
		exit(1)
	end


	Rgsa = Expectation(Pgxiai_xj, Rgxiaixj, [5])

	# 	Qgxa_h = Normalize(rand(dimG, dimX, dimA, dimH, 1, 1, 1), (4,))
	Qgxa_h  = InitEncoderAtBetaInfty_test(dims)

	Iarr = []
	Larr = []
	Lxah = []
	Lcurn = []
	Qxa_h = []
	LnextExpected = []
	# 	BETA = (0.01:0.001:0.1)#(0.01:0.001:0.1)#0.01:0.1:1#:0.5:15
	BETA = (100.0:-10:1)#(0.01:0.001:0.1)#0.01:0.1:1#:0.5:15
	BETA = (1000.0:-100:1)#(0.01:0.001:0.1)#0.01:0.1:1#:0.5:15
	# 	BETA = (1.0:-0.05:0.1)#(0.01:0.001:0.1)#0.01:0.1:1#:0.5:15
	# 	BETA = (1.0:-0.05:0.1)#(0.01:0.001:0.1)#0.01:0.1:1#:0.5:15
	# 	BETA = (1.0:-0.1:0.1)#(0.01:0.001:0.1)#0.01:0.1:1#:0.5:15
	for beta = BETA 
		# 	Qgxa_h = Normalize(rand(dimG, dimX, dimA, dimH, 1, 1, 1), (4,))
		for i = 1:10
			Lcurn, Pxih_ai, LnextExpected, Lxah =  BellmanIB(dims, Pgxiai_xj, Rgsa, Qgxa_h, Pxa_g, Ps_g, beta, gamma)

			# 		println("beta=$(beta), i=$(i)")
			# 		display(reshape(Pxih_ai[1, :, :, 9, 1, 1, 1], (dimX, dimA)))
			# 		println()
			# 		display()

			if false
				Vxh = Expectation(Lxah, Pxih_ai, [3])
				printGridVxh(dims, Vxh, 100, "StateIndpndBeta1000", 1.0)
				exit(1)
			end

			# 		Pxih_ai = Normalize(Pxih_ai .+ 0.05, (3,))
			for j = 1:2
				Qxa_h  = Marginal(JointProb(Qgxa_h, Pxa_g), (1,))
				Qgxa_h = Normalize(broadcast(*, Qxa_h, exp.(-gamma*LnextExpected/beta)), (4, ))
				# 		Qgxa_h = Normalize(broadcast(*, Qxa_h, exp.(-gamma*Lcurn)), (4, ))
			end
			# 		Qgxa_h = Normalize(Qgxa_h .+ 0.00001, (4,))

			# 		Qgxa_h = Normalize(broadcast(*, Qxa_h, exp.(-gamma*Lcurn)), (4, ))
		end
		# 	println("Qxa_h")
		# 	display(Qxa_h)
		# 	println()
		# 	println("Qgxa_h")
		# 	display(Qgxa_h)
		# 	println("LnextExpected:")
		# 	display(LnextExpected)
		# 	println()
		# exit(1)




		# if ~isempty(findall(isnan, Qgxa_h))
		# 	prinln("Nan in Qgxa_h")
		# elseif ~isempty(findall(iszero, Qxa_h))
		# 	prinln("Nan in Qxa_h")
		# elseif ~isempty(findall(isnan, Qxa_h))

		I = Expectation(Pxa_g,Expectation(Qgxa_h, log.(broadcast(/, Qgxa_h, Qxa_h)), [4]), [1])
		push!(Iarr, mean(I))
		push!(Larr, mean(Lcurn))
		println("I[$(beta)]=$(mean(I)), L=$(mean(Lcurn))")
		# 	display(Qxa_h)
		# 	println()
		# 	display(Qgxa_h)
		# 	println()
	end

	fi = figure(1)
	PyPlot.subplot(131)
	PyPlot.plot(BETA, Iarr)
	PyPlot.subplot(132)
	PyPlot.plot(BETA, Larr)
	PyPlot.subplot(133)
	PyPlot.plot(Iarr, Larr)
	savefig(string("./plotsDP/", "BellIB_1.png"), bbox_inches="tight")
	show()
	close(fi)
end

# mainSolveMaze()
### mainSolveMazeGoalCond() ###
### exit(1) ###
### mainIB() ###
# mainEntropyEncoder()
# mainLogValueIterationTest()
# mainEstimatePxa_g()
# mainBellmanIB()

end

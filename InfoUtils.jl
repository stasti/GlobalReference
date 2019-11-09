module InfoUtils

using LinearAlgebra
using ProbUtils

export LocInfo, InitEncoderAtBetaInfty, InitAtBetaInfinityQghC_hN, InitAtBetaInfinityQgxChC_hN, Encoder, EncoderMarginal, InitPxh_gAtBetaInfinity, BA, UpdateMI, IB, EntropyEncoder, ReverseChannel

const debug_InfoUtils = false
const nItersBA = 20
const progressFrac = 5 

# JointProb!(Px_y::Array{Float32}, Px::Array{Float32}, Pxy::Array{Float32}) = broadcast!(*, Pxy, Px_y, Px)
# 

function ReverseChannel!(Px_y::Array{Float32}, Px::Array{Float32})
	Pxy = JointProb(Px_y, Px, Pxy)

		
end



function IB(dims::Tuple, Px_ga::Array{Float64}, beta::Float64)

	dimX1, dimX2, dimA, dimG, dimH = dims
	dimX = dimX1*dimX2

	Px_ga = Normalize(Px_ga .+ 0.001, (1,3))

	Px_g  = Marginal(Px_ga, (3, ))
	Pxg_a = broadcast(/, Px_ga, Px_g)

	Px_h  	= Normalize(rand(1, dimX, 1, dimH, 1, 1, 1), (4,))
	Pxh_a   = Normalize(rand(1, dimX, dimA, dimH, 1, 1, 1), (3,)) 

	Pxg_h = Float64[]
	Pxh_g = Float64[]

	for i in 1:100
		Dxgh   = sum(broadcast(*, Pxg_a, log.(broadcast(/, Pxg_a, Pxh_a))), dims=3)
		Pxg_h = broadcast(*, Px_h, exp.(-beta*Dxgh)) 
		Pxg_h = Normalize(Pxg_h, (4,))
		Px_h  = sum(broadcast(*, Pxg_h, Px_g), dims=1)
		Pxh_g = broadcast(/, broadcast(*, Pxg_h, Px_g), Px_h)
		Pxh_a = sum(broadcast(*, Pxg_a, Pxh_g), dims=1)
	end

	return Pxg_h, Px_h, Pxh_a, Pxh_g
end

function DKL(P::Array{Float64}, Q::Array{Float64})

	nonZeroP = findall(!iszero, P)
	nonZeroQ = findall(!iszero, Q)
	nonZero = intersect(nonZeroP, nonZeroQ)

# 	println("AAA")
# 	println(nonZeroP)
# 	println(nonZeroQ)
# 	println(nonZero)
# 	println("BBB")
# 	exit(1)

# 	return sum(log.( (P[nonZero]./Q[nonZero]).^P[nonZero]))
	return sum(log.( (P).^P)) - sum(log.( (Q).^P))
end

function BA(	dims::Tuple, 
		QxChC_hN::Array{Float64}, 
		Vgxh::Array{Float64}, 
		PgxCaC_xN::Array{Float64}, 
		PxChP_aC::Array{Float64}, 
		Pxh_g::Array{Float64}, gamma::Float64)  
	
	dimX1, dimX2, dimA, dimG, dimH = dims
	dimX = dimX1*dimX2

	QxChC_hN_prev = zeros(size(QxChC_hN))	
	QgxChC_hN = Float64[]#zeros(dimG, dimX, 1, )
	for iter in 1:nItersBA	
# 		if mod(iter, convert(Int, floor(nItersBA/progressFrac))) == 0
# 			println("iterBA[$(iter)]: Dkl[Pj||Pi]=$(DKL(QxChC_hN, QxChC_hN_prev))")
# 			println("iterBA[$(iter)]")
# 		end

		QgxChC_hN = Encoder(dims, QxChC_hN, Vgxh, PgxCaC_xN, PxChP_aC, gamma)  
		QxChC_hN  = EncoderMarginal(QgxChC_hN, Pxh_g)
# 		println("debug BA at $(@__LINE__)")
# 		println(QxChC_hN_prev)
# 		println(QxChC_hN)
# 		println("QxChC_hN-QxChC_hN_prev=\n", QxChC_hN-QxChC_hN_prev)
# 		QxChC_hN_prev = QxChC_hN
# 		TEST_PROB_UNITY(QxChC_hN, "QxChC_hN", "debug BA at $(@__MODULE__), $(@__LINE__)", [7])
# display(QxChC_hN[1, 2, 1, 1, 1, 1, :])
	end

	return QgxChC_hN, QxChC_hN
end

function EncoderArgument(dims::Tuple, Vgxh::Array{Float64}, PgxCaC_xN, Pxh_a::Array{Float64})

	dimX1, dimX2, dimA, dimG, dimH = dims

	dimX = dimX1*dimX2

	Vgxh_next  = permutedims(Vgxh, (1, 5, 3, 7, 2, 6, 4))
	tmp1   = Expectation(Vgxh_next, PgxCaC_xN, [5])
	tmp2   = Expectation(tmp1, Pxh_a, [3])

	if false
		println("debug EncoderArgument at $(@__MODULE__), $(@__LINE__)")
		for hN in 1:dimH, hC in 1:dimH, xi in 1:dimX
			Vg = reshape(tmp2[:, xi, 1, hC, 1, 1, hN], (dimX1, dimX1))
			println("hN=$(hN), hC=$(hC), xi=$(xi)")
			display(Vg)
			println()
		end
# 		exit(1)
	end
	
	return tmp2

end

function Encoder(	dims::Tuple, 
			QxChC_hN::Array{Float64}, 
			Vgxh::Array{Float64}, 
			PgxCaC_xN::Array{Float64}, 
			PxChP_aC::Array{Float64}, gamma::Float64)  

	dimX1, dimX2, dimA, dimG, dimH = dims
	dimX = dimX1*dimX2

	expArg       = EncoderArgument(dims, gamma*Vgxh, PgxCaC_xN, PxChP_aC)

# 	expArgStable = expArg .- maximum(expArg)
	expArgStable = expArg

	if false
		println("debug expArgStable at $(@__MODULE__), $(@__LINE__)")
		for hN in 1:dimH, hC in 1:dimH, xi in 1:dimX
			Vg = reshape(expArgStable[:, xi, 1, hC, 1, 1, hN], (dimX1, dimX1))
			println("hN=$(hN), hC=$(hC), xi=$(xi)")
			display(Vg)
			println()
		end
# 		exit(1)
	end

	QgxChC_hN    = Normalize(broadcast(*, QxChC_hN, exp.(-expArgStable)), (7, ))

# 	println("QxChC_hN=\n", QxChC_hN)	
# 	println("QgxChC_hN=\n", QgxChC_hN)	
# 	exit(1)

	if debug_InfoUtils
		TEST_PROB_UNITY(QgxChC_hN, "QgxChC_hN", "debug Encoder at $(@__MODULE__), $(@__LINE__)", [7])
	end

	return QgxChC_hN
end

function EncoderMarginal(QgxChC_hN::Array{Float64}, Pxh_g::Array{Float64})
	QxChC_hN =  Marginal(JointProb(QgxChC_hN, Pxh_g), (1,))
	if debug_InfoUtils
		TEST_PROB_UNITY(QxChC_hN, "QxChX_hN", "debug EncoderMarginal at $(@__MODULE__), $(@__LINE__)", [7])
	end
	return QxChC_hN
end

function LocInfo(dims::Array{Int}, QgxCaC_hC::Array{Float64}, PxCaC_g::Array{Float64})

	dimX1, dimX2, dimA, dimG, dimH =  dims
	dimX = dimX1*dimX2

	println("START UpdateMI at $(@__FILE__), $(@__LINE__)")

	PxCaC_ghC = JointProb(QgxCaC_hC, PxCaC_g)
	QxCaC_hC  = Marginal(PxCaC_ghC, [1])

	nonZeroIdx = findall(!iszero, PxCaC_ghC)[2]

	mi = zeros(dimG, dimX, dimA, 1, 1, 1, dimH)

	mi[nonZeroIdx] = log2(broadcast(/, QgxCaC_hC, QxCaC_hC)[nonZeroIdx])

	return mi

end

function UpdateMI(dims::Tuple, QgxCaC_hC::Array{Float64}, PxCaC_g::Array{Float64})

	dimX1, dimX2, dimA, dimG, dimH =  dims
	dimX = dimX1*dimX2

	println("START UpdateMI at $(@__FILE__), $(@__LINE__)")

	PxCaC_ghC = Normalize(JointProb(QgxCaC_hC, PxCaC_g) .+ 0.0001, (1, 7))
	QxCaC_hC  = Normalize(Marginal(PxCaC_ghC, (1,)) .+ 0.0001, (7,))

	nonZeroIdx = findall(!iszero, PxCaC_ghC)[2]

# 	MI = zeros(dimX, dimA, dimG, dimH, 1, 1)
# 	mi = zeros(dimX, dimA, dimG, dimH, 1, 1)
	MI = zeros(1,    dimX, 1, dimH, 1, 1,    1)
	mi = zeros(dimG, dimX, 1, dimH, 1, 1, dimH)

	mi[nonZeroIdx] = log(broadcast(/, QgxCaC_hC, QxCaC_hC)[nonZeroIdx])

	MI = Expectation(mi, PxCaC_ghC, [1, 7])

	if length(findall(MI .< 0)) > 0
		println("MI has negative number at UpdateMI($(@__LINE__)")
		exit(1)
	end

	return MI, mi
end

function InitAtBetaInfinityQgxChC_hN(dims::Tuple)

	dimX1, dimX2, dimA, dimG, dimH = dims

	dimX = dimX1*dimX2

	QgxChC_hN = zeros(Float64, dimG, dimX, 1, dimH, 1, 1, dimH)
	
	for hi = 1:dimH, xi =1:dimX
		@inbounds QgxChC_hN[:, xi, 1, hi, 1, 1, :] = Matrix{Float64}(I, dimG, dimH)
	end

	if debug_InfoUtils
		TEST_PROB_UNITY(QgxChC_hN, "QgxChC_hN", "InitAtBetaInfinityQgxChC_hN $(@__LINE__)", [7])
	end

	return QgxChC_hN
end
function InitAtBetaInfinityQghC_hN(dims::Tuple)

	dimX1, dimX2, dimA, dimG, dimH = dims

	dimX = dimX1*dimX2

	QghC_hN = zeros(Float64, dimG, 1, 1, dimH, 1, 1, dimH)
	
	for hi = 1:dimH
		@inbounds QghC_hN[:, 1, 1, hi, 1, 1, :] = Matrix{Float64}(I, dimG, dimH)
	end

	if debug_InfoUtils
		TEST_PROB_UNITY(QghC_hN, "QghC_hN", "InitAtBetaInfinityQghC_hN $(@__LINE__)", [7])
	end

	return QghC_hN
end

# function InitEncoderAtBetaInfty(dims::Array{Int})
function InitEncoderAtBetaInfty(dims::Tuple)

	dimX1, dimX2, dimA, dimG, dimH = dims

	dimX = dimX1*dimX2

	QgxCaC_hC = zeros(Float64, dimG, dimX, dimA, 1, 1, 1, dimH)
	
	for xi = 1:dimX, ai = 1:dimA
		QgxCaC_hC[:, xi, ai, 1, 1, 1, :] = Matrix{Float64}(I, dimG, dimH)
	end

	if debug_InfoUtils
		TEST_PROB_UNITY(QgxCaC_hC, "QgxCaC_hC", "InitEncoderAtBetaInfty", [7])
	end

	return QgxCaC_hC
end

function InitPxh_gAtBetaInfinity(dims::Tuple)
	dimX1, dimX2, dimA, dimG, dimH = dims

	dimX = dimX1*dimX2
	Pxh_g = zeros(dimG, dimX, 1, dimH, 1, 1, 1)
	for xi in 1:dimX
		Pxh_g[:, xi, 1, :, 1, 1, 1] = Matrix{Float64}(I, dimG, dimH)
	end
	return Pxh_g
end

function EntropyEncoder(	dims::Tuple,
				Pxih_ai::Array{Float64},
				Pgxiai_xj::Array{Float64},
				Fxiai::Array{Float64}, gamma::Float64)

	dimX1, dimX2, dimA, dimG, dimH = dims
	dimX = dimX1*dimX2

	Fxjaj   = permutedims(Fxiai,   (1, 5, 6, 4, 2, 3, 7))	
	Pxjh_aj = permutedims(Pxih_ai, (1, 5, 6, 4, 2, 3, 7)) 

	expArg = Expectation(Fxjaj, JointProb(Pgxiai_xj, Pxjh_aj), [5, 6])
	expArgStable = expArg .- maximum(expArg)

	Q = exp.(-gamma*expArgStable)
	Z = sum(Q, dims=7)

	Q = broadcast(/, Q, Z)

	return Q, Z

end

end

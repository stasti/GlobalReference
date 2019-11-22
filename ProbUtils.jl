__precompile__(true)
module ProbUtils

using LinearAlgebra
using Random
using CuArrays

export Expectation!, JointProb!, JointProb, printProb, calcPxiai_xjaj, InitProb, Expectation, TEST_PROB_UNITY, Normalize, Marginal, InitEncoderAtBetaInfty_test, DeltaEncoder, @printMat


macro printMat(expr)
	return :(println(size($expr)); display($expr); println())
end

function printProb(P::Array{Float64, 4}, Str::String)
	println("$(Str)=\n\t", round(P, 3))
end

function DeltaEncoderWithActions(dims::Tuple)

	dimX1, dimX2, dimA, dimG, dimH = dims

	dimX = dimX1*dimX2

	Qgxiai_h = zeros(dimG, dimX, dimA, dimH, 1, 1, 1)

	for xi = 1:dimX, ai = 1:dimA
		Qgxiai_h[:, xi, ai, :, 1, 1, 1] = Matrix{Float64}(I, dimG, dimH)
	end

	# 	Qgxiai_h = Normalize(Qgxiai_h .+ 0.0001, (4,))

	# 	TEST_PROB_UNITY(Qxiaig_h, "Qxiaig_h", "InitEncoderAtBetaInfty", [4])

	return Qgxiai_h
end
function DeltaEncoder(dims::Tuple)

	dimX1, dimX2, dimA, dimG, dimH = dims

	dimX = dimX1*dimX2

	Qgxi_h = zeros(dimG, dimX, 1, dimH, 1, 1, 1)

	if dimH == dimG
		for xi = 1:dimX
			Qgxi_h[:, xi, 1, :, 1, 1, 1] = Matrix{Float64}(I, dimG, dimH)
		end
	else
		random = false
		if random
			for xi = 1:dimX
				for g = 1:dimG
					Qgxi_h[g, xi, 1, shuffle(1:dimH)[1], 1, 1, 1] = 1.0		
				end
			end
		else
			dH = convert(Int, floor(dimG/dimH))
# 			dH = convert(Int, ceil(dimG/dimH))
			for xi = 1:dimX
				for h = 1:dimH 
					Qgxi_h[(h-1)*dH+1:h*dH, xi, 1, h, 1, 1, 1] .= 1.0
				end
			end
		end
	end
# 		display(Qgxi_h[:, 1, 1, :, 1, 1, 1])
# 		exit(1)

	# 	Qgxiai_h = Normalize(Qgxiai_h .+ 0.0001, (4,))

	# 	TEST_PROB_UNITY(Qxiaig_h, "Qxiaig_h", "InitEncoderAtBetaInfty", [4])

	return Qgxi_h
end

function InitEncoderAtBetaInfty_test(dims::Tuple)

	dimX1, dimX2, dimA, dimG, dimH = dims

	dimX = dimX1*dimX2

	Qgxiai_h = zeros(dimG, dimX, dimA, dimH, 1, 1, 1)

	for xi = 1:dimX, ai = 1:dimA
		Qgxiai_h[:, xi, ai, :, 1, 1, 1] = Matrix{Float64}(I, dimG, dimH)
	end


	Qgxiai_h = Normalize(Qgxiai_h .+ 0.0001, (4,))

	# 	TEST_PROB_UNITY(Qxiaig_h, "Qxiaig_h", "InitEncoderAtBetaInfty", [4])

	return Qgxiai_h
end

function Normalize(tP::CuArray{Float32, }, dims::Tuple)
	P = broadcast(/, tP, sum(tP, dims=dims))
	return P
end
function Normalize(tP::Array{BigFloat, }, dims::Tuple)
	P = broadcast(/, tP, sum(tP, dims=dims))
	return P
end
function Normalize(tP::CuArray{Float64, }, dims::Tuple)
	P = broadcast(/, tP, sum(tP, dims=dims))
	return P
end
function Normalize(tP::Array{Float64, }, dims::Tuple)
	P = broadcast(/, tP, sum(tP, dims=dims))
	return P
end
function Normalize(tP::Array{Float32, }, dims::Tuple)
	P = broadcast(/, tP, sum(tP, dims=dims))
	return P
end

function calcPxiai_xjaj(Pxiai_xj::Array{Float64, }, Pxj_aj::Array{Float64, })

	Pxiai_xjaj = broadcast(.*, Pxiai_xj, Pxj_aj)
	return Pxiai_xjaj
end


function JointProb!(TMP::Array{Float64}, Px_y::Array{Float64}, Px::Array{Float64})
	return broadcast!(*, TMP, Px_y, Px)
end
function JointProb(Px_y::Array{Float64,}, Px::Array{Float64,})
	return broadcast(*, Px_y, Px)
end

function TEST_PROB_UNITY(P::Array{Float64, }, str::String, callFrom::String, dims::Array{Int, })

	### Px_y ###
	Psz = size(P)
	ry = sum(P, dims=dims)
	Ry = sum(ry)


	Rx = prod(Psz[setdiff( 1:length(Psz), dims )]) 

	if ~isapprox(Ry, Rx, atol=1e-8)
		println("\ncallFrom: $(callFrom), length $(str)=", length(Psz))
		println("prob $(str)[$(Psz)] is not normalized: Ry=$(Ry) vs Rx=$(Rx)")
		# 		println("ry:\n", ry)
		# 		display(backtrace())
		println("EXIT at ", @__LINE__)
		exit(1)
	else
		println("\ncallFrom: $(callFrom), length $(str)=", length(Psz))
		println("prob $(str)[$(Psz)] is     normalized: Ry=$(Ry) vs Rx=$(Rx)")

	end
end

function Marginal(P::Array{Float64, }, dims::Tuple)
	return sum(P, dims=dims)
end


function InitProb(dims::Tuple, probDims::Tuple, rnd::Bool=true)

	if rnd
		P = reshape(rand(prod(dims)), dims)
	else
		P = reshape(ones(prod(dims)), dims)
	end
	P = broadcast(/, P, sum(P, dims=probDims))
	return P
end

function Expectation!(F::Array{Float64, }, P::Array{Float64, }, ARR1::Array{Float64}, dims::Array{Int, })
	return 	sum(broadcast!(*, ARR1, F, P), dims=dims)
	# 	return Fexpected
end
function Expectation(F::Array{Float64, }, P::Array{Float64, }, dims::Array{Int, })
	Fexpected = sum(broadcast(*, F, P), dims=dims)
	return Fexpected
end



end

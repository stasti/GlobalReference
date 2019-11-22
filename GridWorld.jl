__precompile__(true)


module GridWorld

using ProbUtils
using StatsBase
using LinearAlgebra
using InfoUtils
### sample(["a", "b", "c"], pweights([0.2, 0.5, 0.3]), 10) ###

export SetPxj_xiai, GenerateGoalSet, CreateWalls, InitPxiaig_xj, AvrgStepSize, ActionStrings, EstimatePxa_g, lin2cart, cart2lin, PrintPolicy, ReversePolicy, SetPxj_xiai_new


function ReversePolicy( dims::Tuple, Pxi_ai::Array{Float64} )

	println("\nReversPxi_ai")

	dimX1, dimX2, dimA, dimG, dimH = dims
	dimX = dimX1*dimX2

	Rxi_ai = zeros(dimX, dimA, 1, 1)

	# 			a[1] = "LL"
	# 			a[2] = "DD"
	# 			a[3] = "UU"
	# 			a[4] = "RR" 
	# 			a[5] = "SS"
	P2R = [4, 3, 2, 1, 5]

	for x in 1:dimX	
		idx = findall(!iszero, Pxi_ai[x, :, 1, 1])
		Rxi_ai[x, P2R[idx[1]], 1, 1] = 1.0
	end

	return Rxi_ai

end

function PrintPolicy(dims::Tuple, Pxi_ai::Array{Float64})

	dimX1, dimX2, dimA, dimG, dimH = dims
	dimX = dimX1*dimX2

	a = ActionStrings(dimA, true)

	# 	for x in 1:dimX
	# 		idx = findall(!iszero, Pxi_ai[x, :, 1, 1])[1]
	# 		for ai in 1:dimA
	# 			if ai == idx
	# 				print(a[idx])
	# 			else
	# 				print("\u2218")
	# 			end
	# 		end
	# 		println()
	# 	end

	for x1 in 1:dimX1
		for x2 in 1:(dimX2-1)
			x = cart2lin(dimX1, dimX2, x1, x2)
			# 			for x in 1:dimX
			idx = findall(!iszero, Pxi_ai[x, :, 1, 1])[1]

			print(a[idx])

			# 			end
		end
		x = cart2lin(dimX1, dimX2, x1, dimX2)
		idx = findall(!iszero, Pxi_ai[x, :, 1, 1])[1]
		println(a[idx])
	end
end


function EstimatePxa_g(dims::Tuple, Pgxiai_xj::Array{Float64}, Pgxi_ai::Array{Float64})

	dimX1, dimX2, dimA, dimG, dimH = dims
	dimX = dimX1*dimX2

	gsaCnts = ones(dimG, dimX, dimA, 1, 1, 1, 1)
	aStay = 5
	@time for gi = repeat(1:dimG, 50)
		for x0 = repeat(1:dimX, 50)
			if x0 == gi
				gsaCnts[gi, x0, aStay, 1, 1, 1, 1] = gsaCnts[gi, x0, aStay, 1, 1, 1, 1] + 1
				continue
			end
			for trj = 1:50
				xi = x0
				done = false 
				while ~done
					# 				for t = 1:50
					ai = sample(1:dimA, pweights(Pgxi_ai[gi, xi, :, 1, 1, 1, 1]))
					xj = sample(1:dimX, pweights(Pgxiai_xj[gi, xi, ai, 1, :, 1, 1]))
					if xj == gi
						done = true
					end
					gsaCnts[gi, xi, ai, 1, 1, 1, 1] = gsaCnts[gi, xi, ai, 1, 1, 1, 1] + 1
					xi = xj
				end
			end
		end
	end

	Pgsa = Normalize(gsaCnts .+ 1, (1, 2, 3))

	return Pgsa
end

function find(A::Array{Int64, })
	return findall(!iszero, A)
end

# function find(A::Array{Float64, })
# 	return findall(!iszero, A)
# end

function countnz(A::Array{Int64, })
	return length(findall(!iszero, A))
end

function lin2cart(dimX1::Int, dimX2::Int, linIdx::Int)
	n::Int = ceil(linIdx/dimX1)
	j::Int = n
	i::Int  = dimX1 - (n*dimX1 - linIdx)
	return i, j
end

function cart2lin(dimX1::Int, dimX2::Int, i::Int, j::Int)

	if Bool(i>dimX1) || Bool(j>dimX2) || i < 1 || j < 1
		return -1
	end

	return dimX1*(j-1) + i 
end


function ActionStrings(dimA::Int, unicode::Bool=false)

	a = Array{String}(undef, dimA, 1)
	if dimA == 9
		a[1] = "DL"
		a[2] = "LL"
		a[3] = "UL"
		a[4] = "DD"
		a[5] = "UU"
		a[6] = "DR"
		a[7] = "RR" 
		a[8] = "UR"
		a[9] = "SS"
	end
	if dimA == 5
		if unicode == false
			a[1] = "LL"
			a[2] = "DD"
			a[3] = "UU"
			a[4] = "RR" 
			a[5] = "SS"
		else
			a[1] = "\u2190"
			a[2] = "\u2193"
			a[3] = "\u2191"
			a[4] = "\u2192" 
			a[5] = "\u21BB"
		end
	end
	return a
end

function Actions(dimA::Int)
	a = zeros(Int, dimA, 2)
	if dimA == 9
		a[1, :] = [+1, -1]
		a[2, :] = [ 0, -1]
		a[3, :] = [-1, -1]
		a[4, :] = [+1,  0]
		a[5, :] = [-1,  0]
		a[6, :] = [+1, +1]
		a[7, :] = [ 0, +1]
		a[8, :] = [-1, +1]
		a[9, :] = [ 0,  0]
	end
	if dimA == 5
		a[1, :] = [ 0, -1]
		a[2, :] = [+1,  0]
		a[3, :] = [-1,  0]
		a[4, :] = [ 0, +1]
		a[5, :] = [ 0,  0]
	end
	if dimA == 4
		a[1, :] = [ 0, -1]
		a[2, :] = [+1,  0]
		a[3, :] = [-1,  0]
		a[4, :] = [ 0, +1]
		# 		a[5, :] = [ 0,  0]
	end
	return a
end

function SetPxj_xiai_new(dims::Tuple, actionProb::Float64, C::Array{Int})

	dimX1, dimX2, dimA, dimG, dimH = dims
	dimX = dimX1*dimX2

	BuildMazeTransitionProb_DEBUG = false

	epsilon    = eps(Float64)
	dimX       = dimX1*dimX2
	Pxiai_xj   = zeros(Float64, dimX, dimA, dimX, 1)
	actions    = Actions(dimA)
	actionStr  = ActionStrings(dimA)
	# 	endCost  = 0.0
	endCost  = +0.5#0.5
	endCost  = +0.0#0.5
	# 	endCost  = #0.5
	wallCost = -1000000.0
	cost     = -1.0 
	# 	cost     = -0.0 
	R        = cost*ones(Int, dimX, dimA, dimX, 1)
	Xf = -1
	stayAction = 5
	for xi = 1:dimX


		crnt_i, crnt_j = lin2cart(dimX1, dimX2, xi)

		if C[crnt_i, crnt_j] == 0
			continue
		end
	
		nextStates     = ValidNeighborhood(dims, xi, C) 
		numNextValid   = countnz(nextStates)

		if xi == Xf && false # no absorbing state
			R[xi, :, Xf] .= endCost
			# 			R[xi, stayAction, Xf] = 0.0
			### absorbing ###
			Pxiai_xj[xi, :, Xf] .= 1.0

			### not absorbing ###
			# 			Pxiai_xj[xi, stayAction, Xf] = 1.0
			continue
		end

		for ai = 1:dimA

# 			println("xi[$(xi)], $(actionStr[ai])")
			next_i, next_j = crnt_i+actions[ai, 1], crnt_j+actions[ai, 2]

			xj = cart2lin(dimX1, dimX2, next_i, next_j)
			nextByAction = isValidNeighborhood(dims, C, next_i, next_j)
# 			display(reshape(nextByAction, 3, 3));  println();

			if countnz(nextByAction) == 0 # action is not valid
				if true # last working 
					Pxiai_xj[xi, ai, xi] = 1.0
				else
					# 					Pxiai_xj[xi, ai, xi] = 0.8
					# 					Pxiai_xj[xi, ai, setdiff(1:dimX, xi)].=(1-0.8)/(length(setdiff(1:dimX,xi)))
					Pxiai_xj[xi, ai, :] .= 1/dimX 
				end
				if Bool(next_i>0) && Bool(next_j>0) && xj > 0  
					R[xi, ai, xj] = wallCost
				end
			else
				Pxiai_xj[xi, ai, findall(!iszero, nextByAction)] .= actionProb

				test_diff = true
				if test_diff
					nextByAction_j = []
					for aj = 1:dimA
						if aj == ai
							continue
						end
						z_i, z_j = crnt_i+actions[aj, 1], crnt_j+actions[aj, 2]
						z_lin    = cart2lin(dimX1, dimX2, z_i, z_j)
						if 0<z_i<=dimX1 && 0<z_j<=dimX2
							if C[z_i, z_j] == 1
								push!(nextByAction_j, z_lin)
							end
						end

					end
					
# 					tmp = sum(findall(!iszero, nextByAction_j))
# 					display(nextByAction_j)
# 					display(nextByAction)
for k in 1:length(nextByAction_j)
	Pxiai_xj[xi, ai, nextByAction_j[k]] = (1-actionProb)/(length(nextByAction_j))  
end


				end#test_diff



				if false
					Pxiai_xj[xi, ai, findall(iszero, nextByAction)] .= (1-actionProb)/(dimX-1)  
				end
			end
		end
	end


# 	display(sum(Pxiai_xj, dims=3))

# 	for xi = 1:dimX
# 		for ai in 1:dimA
# 			println("x[$(xi)], a[$(actionStr[ai])]")
# 			display(reshape(Pxiai_xj[xi, ai, :], (dimX1, dimX2))); println()
# 		end
# 	end
# 
# 	exit(1)
	return Pxiai_xj, R
end
function SetPxj_xiai(dims::Tuple, actionProb::Float64, C::Array{Int})

	dimX1, dimX2, dimA, dimG, dimH = dims
	dimX = dimX1*dimX2

	BuildMazeTransitionProb_DEBUG = false

	epsilon    = eps(Float64)
	dimX       = dimX1*dimX2
	Pxj_xiai   = zeros(Float64, dimX, dimA, dimX, 1)
	actions    = Actions(dimA)
	actionStr  = ActionStrings(dimA)
	# 	endCost  = 0.0
	endCost  = +0.5#0.5
	endCost  = +0.0#0.5
	# 	endCost  = #0.5
	wallCost = -1000000.0
	cost     = -1.0 
	# 	cost     = -0.0 
	R        = cost*ones(Int, dimX, dimA, dimX, 1)
	Xf = -1
	stayAction = 5
	for xi = 1:dimX

		crnt_i, crnt_j = lin2cart(dimX1, dimX2, xi)

		nextStates     = ValidNeighborhood(dims, xi, C) 
		numNextValid   = countnz(nextStates)

		if xi == Xf && false # no absorbing state
			R[xi, :, Xf] .= endCost
			# 			R[xi, stayAction, Xf] = 0.0
			### absorbing ###
			Pxj_xiai[xi, :, Xf] .= 1.0

			### not absorbing ###
			# 			Pxj_xiai[xi, stayAction, Xf] = 1.0
			continue
		end

		for ai = 1:dimA

# 			println("xi[$(xi)], $(actionStr[ai])")
			next_i, next_j = crnt_i+actions[ai, 1], crnt_j+actions[ai, 2]

			xj = cart2lin(dimX1, dimX2, next_i, next_j)
			nextByAction = isValidNeighborhood(dims, C, next_i, next_j)
# 			display(reshape(nextByAction, 3, 3));  println();

			if countnz(nextByAction) == 0 # action is not valid
				if true # last working 
					Pxj_xiai[xi, ai, xi] = 1.0
				else
					# 					Pxj_xiai[xi, ai, xi] = 0.8
					# 					Pxj_xiai[xi, ai, setdiff(1:dimX, xi)].=(1-0.8)/(length(setdiff(1:dimX,xi)))
					Pxj_xiai[xi, ai, :] .= 1/dimX 
				end
				if Bool(next_i>0) && Bool(next_j>0) && xj > 0  
					R[xi, ai, xj] = wallCost
				end
			else
				# 				Pxj_xiai[xi, ai, find(vec(nextByAction))] .= 1.0  
				Pxj_xiai[xi, ai, findall(!iszero, nextByAction)] .= actionProb

				if false
					Pxj_xiai[xi, ai, findall( iszero, nextByAction)] .= (1-actionProb)/(dimX-1)  
					# 				else

				end
			end
		end

	end
# exit(1)
	return Pxj_xiai, R
end

function SetPxj_xiai(dims::Tuple, actionProb::Float64, C::Array{Int, }, Xf::Int)

# 	println("START: SetPxj_xiai Xf=$(Xf)")

	dimX1, dimX2, dimA, dimG, dimH = dims
	dimX = dimX1*dimX2

	BuildMazeTransitionProb_DEBUG = false

	epsilon    = eps(Float64)
	dimX       = dimX1*dimX2
	Pxj_xiai   = zeros(Float64, dimX, dimA, dimX, 1)

	actions    = Actions(dimA)
	actionStr  = ActionStrings(dimA)
	# 	endCost  = 0.0
# 	endCost  = +0.5#0.5
	endCost  = +0.0#0.5
	# 	endCost  = #0.5
	wallCost = -1000000.0
	cost     = -1.0 
	# 	cost     = -0.0 
	R        = cost*ones(Int, dimX, dimA, dimX, 1)

	stayAction = length(actions)

	for xi = 1:dimX

		crnt_i, crnt_j = lin2cart(dimX1, dimX2, xi)

		nextStates     = ValidNeighborhood(dims, xi, C) 
		numNextValid   = countnz(nextStates)

		if xi == Xf
			R[xi, :, Xf] .= endCost

			# 			R[xi, stayAction, Xf] = 0.0
			### absorbing ###
			Pxj_xiai[xi, :, Xf] .= 1.0

			if C[crnt_i, crnt_j] == 0
				R[xi, :, Xf] .= wallCost
			Pxj_xiai[xi, :, Xf] .= 0.0
			end

			### not absorbing ###
			# 			Pxj_xiai[xi, stayAction, Xf] = 1.0
			continue
		end
biasDown = false

		for ai = 1:dimA

			if biasDown
				for aj = 2 # action down bias
					next_i, next_j = crnt_i+actions[aj, 1], crnt_j+actions[aj, 2]

					xj = cart2lin(dimX1, dimX2, next_i, next_j)
					nextByAction = isValidNeighborhood(dims, C, next_i, next_j)

					if countnz(nextByAction) != 0 # action is not valid
						Pxj_xiai[xi, aj, findall(!iszero, nextByAction)] .= 0.2
					end
				end
			end # action down bias

			next_i, next_j = crnt_i+actions[ai, 1], crnt_j+actions[ai, 2]

			xj = cart2lin(dimX1, dimX2, next_i, next_j)
			nextByAction = isValidNeighborhood(dims, C, next_i, next_j)

			if countnz(nextByAction) == 0 # action is not valid
				if true # last working 
					Pxj_xiai[xi, ai, xi] = 1.0
				else
					Pxj_xiai[xi, ai, :] .= 1/dimX 
				end
				if Bool(next_i>0) && Bool(next_j>0) && xj > 0  
					R[xi, ai, xj] = wallCost
				end
			else
				Pxj_xiai[xi, ai, findall(!iszero, nextByAction)] .= actionProb
				Pxj_xiai[xi, ai, findall( iszero, nextByAction)] .= (1-actionProb)/(dimX-1)  
			end
		end


	end

	Pxj_xiai = Normalize(Pxj_xiai, (3, ))

	return Pxj_xiai, R
end

function AvrgStepSize(dims::Tuple, V::Array{Float64})

	dimX1, dimX2, dimA, dimG, dimH = dims
	dimX = dimX1*dimX2

	totNumSteps = (dimX1-1)*dimX2 + (dimX2-1)*dimX1
	avrgStepSize = 0
	for xi in 1:dimX
		V_d1Xd2 = reshape(V[1, xi, 1, :, 1, 1, 1], dimX1, dimX2)
		avrgStepSize += (sum(abs.(diff(V_d1Xd2, dims=1))) + sum(abs.(diff(V_d1Xd2, dims=2))))/totNumSteps
	end

	return avrgStepSize/dimX

end

function InitPxiaig_xj(dims::Tuple, G::Array{Int, }, C::Array{Int, }, actionProb)

	dimX1, dimX2, dimA, dimG, dimH = dims

	dimX = dimX1*dimX2

	println(dims)

	# 	Pxiaig_xj = Array{Float64}(undef, dimX, dimA, dimG, 1, dimX, 1)
	# 	Rxiaigxj  = Array{Float64}(undef, dimX, dimA, dimG, 1, dimX, 1)

	# 	Pxiaig_xj = Array{Float64}(undef, dimG, dimX, dimA, 1, dimX, 1, 1)
	# 	Rxiaigxj  = Array{Float64}(undef, dimG, dimX, dimA, 1, dimX, 1, 1)
	Pxiaig_xj = Array{Float64}(undef, dimG, dimX, dimA, dimX, 1)
	Rxiaigxj  = Array{Float64}(undef, dimG, dimX, dimA, dimX, 1)

	for gi in 1:dimG
		Pxiaig_xj[gi, :, :, :, :],
		Rxiaigxj[gi, :, :, :, :] = SetPxj_xiai(dims, actionProb, C, gi)
# 		Rxiaigxj[gi, :, :, :, :] = SetPxj_xiai(dims, actionProb, C, G[gi])
		# 		Rxiaigxj[gi, :, :, 1, :, 1, 1] = SetPxj_xiai(dims, actionProb, C)
	end

	return Pxiaig_xj, Rxiaigxj
end

# function GenerateGoalSet(dims::Array{Int,}, goalType::String, walls::Array{Int,})
function GenerateGoalSet(dims::Tuple, goalType::String, walls::Array{Int,})
	dimX1, dimX2 = dims[1:2]
	dimX = dimX1*dimX2
	if goalType == "fullX"#.e.g., 1:dimX
		Gtmp = collect(1:dimX) 
	elseif goalType == "singleX"#.e.g., 23 
		X = parse(Int, match(r"[0-9]+", str, 1).match)
		if X > dimX
			println("G=$(X), while dimX=$(dimX)")
			exit(1)
		end
	elseif goalType == "randomN"#e.g., rand
		X 	= parse(Int, match(r"[0-9]+", str, 1).match)
		if X > dimX
			println("G=$(X), while dimX=$(dimX)")
			exit(1)
		end
		Gtmp 	= shuffle(1:dimX)[1:X]
	elseif goalType == "evenN"#e.g., 1:5:dimX
		X 	= parse(Int, match(r"[0-9]+", str, 1).match)
		if X > dimX
			println("G=$(X), while dimX=$(dimX)")
			exit(1)
		end
		stride = convert(Int, ceil(dimX/X))
		Gtmp = collect(stride:stride:dimX)
	elseif goalType == "contiguousN" #e.g., 17:25 
		X 	= parse(Int, match(r"[0-9]+", str, 1).match)
		Gtmp = collect(1:X)
	end

	G = setdiff(Gtmp, findall(iszero, walls[:])) #goal can not be on the wall

	if isempty(G)
		println("Goal set is empty. all the goals are on the walls")
		exit(1)
	end


	println("$(goalType), G=$(G), len=$(length(G))")
	# 	return G, length(G)
	return G
end

function WallsTypeZ(dimX1::Int, dimX2::Int)

	dimX = dimX1*dimX2
	C    = ones(Int64, dimX1, dimX2)
	width     = 0#floor((dimX1)/20)

	wall_i  = floor((dimX1)/3)
	wall_j  = ceil(2/3*dimX2)
	C[wall_i-width:wall_i+width, 1:wall_j+2] = 0


	wall_i  = ceil(2/3*(dimX1)) + 1
	wall_j  = floor(1/3*dimX2)
	C[wall_i-width:wall_i+width, wall_j:dimX2] = 0

	vert_wall_length = ceil((1/6)*(dimX1))  
	C[wall_i+width - vert_wall_length:wall_i-width, wall_j] = 0
	C[wall_i+width - vert_wall_length, wall_j:dimX2 - floor((1/6)*dimX2)]        = 0

	C[wall_i, wall_j + floor((1/5)*dimX2)] = 1

	if usePyPlot && false # checked
		figure
		PyPlot.matplotlib[:rc]("text", usetex=true)
		fig = PyPlot.imshow(C, cmap="bone", interpolation="none")
		colorbar(fig, ticks=[minimum(C), maximum(C)])
		show()
		exit(1)
	end

	return C

end

function WallsTypeK(dimX1::Int, dimX2::Int)
	dimX = dimX1*dimX2
	C    = ones(Int64, dimX1, dimX2)
	width     = 0#floor((dimX1)/20)

	wall_i  = convert(Int, floor((dimX1)/3))
	wall_j  = convert(Int, ceil(2/3*dimX2))
	C[wall_i-width:wall_i+width, 1:wall_j] .= 0


	wall_i  = convert(Int,ceil(2/3*(dimX1))) + 1
	wall_j  = convert(Int,floor(1/3*dimX2))
	C[wall_i-width:wall_i+width, wall_j:dimX2] .= 0

	vert_wall_length = convert(Int,ceil((1/6)*(dimX1)))  
	C[wall_i+width - vert_wall_length:wall_i-width, wall_j] .= 0
	C[wall_i+width - vert_wall_length, wall_j:dimX2 - convert(Int,floor((1/6)*dimX2))]        .= 0

	C[wall_i, wall_j + convert(Int,floor((1/5)*dimX2))] = 1

	# 	if usePyPlot && false # checked
	# 		figure
	# 		PyPlot.matplotlib[:rc]("text", usetex=true)
	# 		fig = PyPlot.imshow(C, cmap="bone", interpolation="none")
	# 		colorbar(fig, ticks=[minimum(C), maximum(C)])
	# 		show()
	# 		exit(1)
	# 	end

	return C

end

function WallsTypeM(dimX1::Int, dimX2::Int)
	dimX = dimX1*dimX2
	C    = ones(Int64, dimX1, dimX2)
	width     = 0#floor((dimX1)/20)

	wall_i  = floor((dimX1)/3)
	wall_j  = ceil(2/3*dimX2)
	C[wall_i-width:wall_i+width, 1:wall_j] = 0


	wall_i  = ceil(2/3*(dimX1)) + 1
	wall_j  = floor(1/3*dimX2)
	C[wall_i-width:wall_i+width, wall_j:dimX2] = 0

	vert_wall_length = ceil((1/6)*(dimX1))  
	C[wall_i+width - vert_wall_length:wall_i-width, wall_j] = 0
	C[wall_i+width - vert_wall_length, wall_j:dimX2 - floor((1/6)*dimX2)]        = 0

	C[wall_i, wall_j + floor((1/5)*dimX2)] = 1

	C[wall_i-vert_wall_length:wall_i, wall_j-2] = 0

	if usePyPlot && false # checked
		figure
		PyPlot.matplotlib[:rc]("text", usetex=true)
		fig = PyPlot.imshow(C, cmap="bone", interpolation="none")
		colorbar(fig, ticks=[minimum(C), maximum(C)])
		show()
		exit(1)
	end

	return C

end

function WallsTypeBlocks(dimX1::Int, dimX2::Int)

	N = 3
	dimX = dimX1*dimX2
	C    = ones(Int64, dimX1, dimX2)
	
	dX = convert(Int, floor(dimX2/N))
	dX1 = convert(Int, floor(dX/2))

	dY = convert(Int, floor(dimX2/N))
	dY1 = convert(Int, floor(dY/2))

	x_c = range(dX1, step=dX, stop=dimX2-dX1)
	y_c = range(dY1, step=dY, stop=dimX1-dY1)

# println(x_c)
# println(y_c)

	for xi in x_c, yi in y_c 
		C[xi-2:xi+2, yi-2:yi+2] .= 0
	end

	return C

end

function WallsTypeG(dimX1::Int, dimX2::Int)
	dimX = dimX1*dimX2
	C    = ones(Int64, dimX1, dimX2)
	width     = 0#floor((dimX1)/20)

	wall_i  = floor((dimX1)/3)
	wall_j  = ceil(2/3*dimX2)
	C[wall_i-width:wall_i+width, 1:wall_j] = 0


	wall_i  = ceil(2/3*(dimX1)) + 1
	wall_j  = floor(1/3*dimX2)
	C[wall_i-width:wall_i+width, wall_j:dimX2] = 0

	vert_wall_length = ceil((1/6)*(dimX1))  
	C[wall_i+width - vert_wall_length:wall_i-width, wall_j] = 0
	C[wall_i+width - vert_wall_length, wall_j:dimX2 - floor((1/6)*dimX2)]        = 0

	if usePyPlot && false # checked
		figure
		PyPlot.matplotlib[:rc]("text", usetex=true)
		fig = PyPlot.imshow(C, cmap="bone", interpolation="none")
		colorbar(fig, ticks=[minimum(C), maximum(C)])
		show()
	end

	return C

end

function WallsTypeH(dimX1::Int, dimX2::Int)
	dimX = dimX1*dimX2
	C    = ones(Int64, dimX1, dimX2)
	width     = 0#floor((dimX1)/20)

	wall_i = floor((dimX1+1)/2)
	wall_j = floor((dimX2+1)/2)
	C[wall_i, 1:floor((dimX2+1)/3)] = 0
	C[wall_i, floor(2/3*(dimX2+1)):dimX2] = 0

	C[1:floor((dimX1+1)/3), wall_j] = 0
	C[floor(2/3*(dimX1+1)):dimX1, wall_j] = 0

	if usePyPlot && false # checked
		figure
		PyPlot.matplotlib[:rc]("text", usetex=true)
		fig = PyPlot.imshow(C, cmap="bone", interpolation="none")
		colorbar(fig, ticks=[minimum(C), maximum(C)])
		show()
	end

	return C
end

function WallsTypeF(dimX1::Int, dimX2::Int)
	dimX = dimX1*dimX2
	C    = ones(Int64, dimX1, dimX2)
	width     = 0#floor((dimX1)/20)

	wall_i = convert(Int, floor((dimX1)/3))
	wall_j = convert(Int, ceil(2/3*dimX2))
	C[wall_i-width:wall_i+width, 1:wall_j] .= 0


	wall_i = convert(Int, ceil(2/3*(dimX1))) + 1
	wall_j = convert(Int, floor(1/3*dimX2))
	C[wall_i-width:wall_i+width, wall_j:dimX2] .= 0


	return C
end


function WallsTypeRandomHrzn(dimX1, dimX2)

	C    = ones(Int64, dimX1, dimX2)

	nWalls = convert(Int, floor(0.75*dimX1))


	for n in 1:nWalls
		start = (rand(1:dimX1, 1)[1], rand(1:dimX2, 1)[1])
		len   = rand(1:convert(Int, floor(0.5*dimX1)), 1)[1]
		dir   = rand((2, 4), 1)[1]


		# 	start = (9, 9)
		# 	dir = 4
		# 	len= 3

		# 		println((start, len, dir))
		if dir == 1 #up
			len = start[1]-len >= 0 ? len : start[1]  
			# 			println(len)
			# 			println( start[1]:-1:(start[1]-len) )
			C[start[1]:-1:(start[1]-len+1), start[2]] .= 0
		elseif dir == 2 #right
			len = start[2]+len <= dimX1 ? len : dimX1 - start[2]  
			C[start[1], start[2]:(start[2]+len-1)] .= 0
		elseif dir == 3 # down
			len = start[1]+len <= dimX2 ? len : dimX2 - start[1]  
			C[start[1]:(start[1]+len-1), start[2]] .= 0
		elseif dir == 4 # left
			len = start[2]-len >= 0 ? len : start[2]  
			C[start[1], start[2]:-1:(start[2]-len+1)] .= 0
		end
	end

	return C
end
function WallsTypeRandomVert(dimX1, dimX2)

	C    = ones(Int64, dimX1, dimX2)

	nWalls = convert(Int, floor(0.75*dimX1))


	for n in 1:nWalls
		start = (rand(1:dimX1, 1)[1], rand(1:dimX2, 1)[1])
		len   = rand(1:convert(Int, floor(0.5*dimX1)), 1)[1]
		dir   = rand((1, 3), 1)[1]


		# 	start = (9, 9)
		# 	dir = 4
		# 	len= 3

		# 		println((start, len, dir))
		if dir == 1 #up
			len = start[1]-len >= 0 ? len : start[1]  
			# 			println(len)
			# 			println( start[1]:-1:(start[1]-len) )
			C[start[1]:-1:(start[1]-len+1), start[2]] .= 0
		elseif dir == 2 #right
			len = start[2]+len <= dimX1 ? len : dimX1 - start[2]  
			C[start[1], start[2]:(start[2]+len-1)] .= 0
		elseif dir == 3 # down
			len = start[1]+len <= dimX2 ? len : dimX2 - start[1]  
			C[start[1]:(start[1]+len-1), start[2]] .= 0
		elseif dir == 4 # left
			len = start[2]-len >= 0 ? len : start[2]  
			C[start[1], start[2]:-1:(start[2]-len+1)] .= 0
		end
	end

	return C
end
function WallsTypeFFF(dimX1, dimX2)

	C    = ones(Int64, dimX1, dimX2)

	nWalls = convert(Int, floor(0.25*dimX1))
	dWall  = convert(Int, floor(dimX1/nWalls))	

	for n in 1:nWalls-1

		start = (n*dWall, n % 2 == 0 ? dimX2 : 1)
		len   = convert(Int, floor(0.85*dimX2))
		dir   = n % 2 == 0 ? 4 : 2

		if dir == 1 #up
			len = start[1]-len >= 0 ? len : start[1]  
			C[start[1]:-1:(start[1]-len+1), start[2]] .= 0
		elseif dir == 2 #right
			len = start[2]+len <= dimX2 ? len : dimX2 - start[2]  
			C[start[1], start[2]:(start[2]+len-1)] .= 0
		elseif dir == 3 # down
			len = start[1]+len <= dimX1 ? len : dimX1 - start[1]  
			C[start[1]:(start[1]+len-1), start[2]] .= 0
		elseif dir == 4 # left
			len = start[2]-len >= 0 ? len : start[2]  
			C[start[1], start[2]:-1:(start[2]-len+1)] .= 0
		end
	end

	return C
end

function WallsTypeGlobalRef5x5r9(dimX1, dimX2)

	C    = ones(Int64, dimX1+1, dimX2+1)
	dW   = convert(Int, (dimX1 + 2) / 3)

	C[dW:dW:dimX1, :] .= 0
	C[:, dW:dW:dimX2] .= 0

	dH = convert(Int, (((dimX1 - 1) / 3) + 1)/2 )
	dZ = convert(Int, (((dimX1 - 1) / 3) + 1)   ) 

	C[dH:dZ:dimX1, :] .= 1
	C[:, dH:dZ:dimX2] .= 1

	return C
end
function WallsTypeFF(dimX1, dimX2)

	C    = ones(Int64, dimX1, dimX2)


	nWalls = convert(Int, floor(0.25*dimX1))
	dWall  = convert(Int, floor(dimX1/nWalls))	



	for n in 1:nWalls-1

		start = (n*dWall, n % 2 == 0 ? dimX2 : 1)
		len   = convert(Int, floor(0.85*dimX2))
		dir   = n % 2 == 0 ? 4 : 2

		if dir == 1 #up
			len = start[1]-len >= 0 ? len : start[1]  
			C[start[1]:-1:(start[1]-len+1), start[2]] .= 0
		elseif dir == 2 #right
			len = start[2]+len <= dimX2 ? len : dimX2 - start[2]  
			C[start[1], start[2]:(start[2]+len-1)] .= 0
		elseif dir == 3 # down
			len = start[1]+len <= dimX1 ? len : dimX1 - start[1]  
			C[start[1]:(start[1]+len-1), start[2]] .= 0
		elseif dir == 4 # left
			len = start[2]-len >= 0 ? len : start[2]  
			C[start[1], start[2]:-1:(start[2]-len+1)] .= 0
		end
	end

	return C
end

function WallsTypeRandom(dimX1, dimX2)

	C    = ones(Int64, dimX1, dimX2)

	nWalls = convert(Int, floor(0.5*dimX1))


	for n in 1:nWalls
		start = (rand(1:dimX1, 1)[1], rand(1:dimX2, 1)[1])
		len   = rand(1:convert(Int, floor(0.5*dimX1)), 1)[1]
		dir   = rand(1:4, 1)[1]


		# 	start = (9, 9)
		# 	dir = 4
		# 	len= 3

		# 		println((start, len, dir))
		if dir == 1 #up
			len = start[1]-len >= 0 ? len : start[1]  
			# 			println(len)
			# 			println( start[1]:-1:(start[1]-len) )
			C[start[1]:-1:(start[1]-len+1), start[2]] .= 0
		elseif dir == 2 #right
			len = start[2]+len <= dimX2 ? len : dimX2 - start[2]  
			C[start[1], start[2]:(start[2]+len-1)] .= 0
		elseif dir == 3 # down
			len = start[1]+len <= dimX1 ? len : dimX1 - start[1]  
			C[start[1]:(start[1]+len-1), start[2]] .= 0
		elseif dir == 4 # left
			len = start[2]-len >= 0 ? len : start[2]  
			C[start[1], start[2]:-1:(start[2]-len+1)] .= 0
		end
	end

	return C
end
# 

function WallsTypeE(dimX1::Int, dimX2::Int)
	dimX = dimX1*dimX2
	C    = ones(Int64, dimX1, dimX2)
	width    = 0#floor((dimX1)/20)

	wall_i  = floor((dimX1)/3)
	wall_j  = ceil(2/3*dimX2)
	C[wall_i-width:wall_i+width, 1:wall_j] = 0

	if usePyPlot && false # checked
		figure
		PyPlot.matplotlib[:rc]("text", usetex=true)
		fig = PyPlot.imshow(C, cmap="bone", interpolation="none")
		colorbar(fig, ticks=[minimum(C), maximum(C)])
		show()
		exit(1)
	end

	return C
end

function WallsTypeD(dimX1::Int, dimX2::Int)
	dimX = dimX1*dimX2
	C    = ones(Int64, dimX1, dimX2)

	wall_i  = floor((dimX1+1)/2)
	wall_j  = floor((dimX2+1)/2)
	C[1:wall_i, 1:wall_j] = 0

	return C
end
function WallsTypeC(dimX1::Int, dimX2::Int)
	dimX = dimX1*dimX2
	C    = ones(Int64, dimX1, dimX2)

	wall_i  = floor((dimX1+1)/2)
	wall_j  = floor((dimX2+1)/2)
	C[wall_i, wall_j] = 0

	return C
end
function WallsTypeA(dimX1::Int, dimX2::Int)
	dimX = dimX1*dimX2
	C    = ones(Int64, dimX1, dimX2)

	wall1i  = 4
	wall2i  = 9

	for wall1j  = 4:dimX2
		C[wall1i, wall1j] = 0
	end

	for wall2j  = 1:dimX2-3
		C[wall2i, wall2j] = 0
	end
	return C
end

function WallsTypeB(dimX1::Int, dimX2::Int)

	println("CREATED: WallsTypeB at $(@__LINE__)")

	dimX = dimX1*dimX2
	C    = ones(Int64, dimX1, dimX2)

	return C
end

function WallsTypeTest3x3(dimX1::Int, dimX2::Int)

	println("CREATED: WallsTypeTest3x3 at $(@__LINE__)")

	dimX = dimX1*dimX2
	C    = ones(Int64, dimX1, dimX2)
	C[2, 1:2] .= 0

	return C
end

# function CreateWalls(dims::Array{Int,}, wallTypeArg::String)
function CreateWalls(dims::Tuple, wallTypeArg::String)

	dimX1, dimX2 = dims[1:2]

	if wallTypeArg == "B"
		wallType = "WallB"
		C     = WallsTypeB(dimX1, dimX2)
	elseif wallTypeArg == "F" 
		wallType = "WallF"
		C     = WallsTypeF(dimX1, dimX2)
	elseif wallTypeArg == "K"
		wallType = "WallK"
		C     = WallsTypeK(dimX1, dimX2)
	elseif wallTypeArg == "H"
		wallType = "WallH"
		C     = WallsTypeH(dimX1, dimX2)
	elseif wallTypeArg == "G"
		wallType = "WallG"
		C     = WallsTypeG(dimX1, dimX2)
	elseif wallTypeArg == "M"
		wallType = "WallM"
		C     = WallsTypeM(dimX1, dimX2)
	elseif wallTypeArg == "Z"
		wallType = "WallZ"
		C     = WallsTypeZ(dimX1, dimX2)
	elseif wallTypeArg == "Random"
		wallType = "WallRandom"
		println("wallRandom")
		C     = WallsTypeRandom(dimX1, dimX2)
	elseif wallTypeArg == "RandomVert"
		wallType = "WallRandomVert"
		println("wallRandomVert")
		C     = WallsTypeRandomVert(dimX1, dimX2)
	elseif wallTypeArg == "RandomHrzn"
		wallType = "WallRandomHrzn"
		println("wallRandomHrzn")
		C     = WallsTypeRandomHrzn(dimX1, dimX2)
	elseif wallTypeArg == "FF"
		wallType = "WallFF"
		println("wallFF")
		C     = WallsTypeFF(dimX1, dimX2)
	elseif wallTypeArg == "Blocks"
		wallType = "WallBlocks"
		println("wallBlocks")
		C     = WallsTypeBlocks(dimX1, dimX2)
	elseif wallTypeArg == "Test3x3"
		wallType = "WallTest3x3"
		println(wallType)
		C     = WallsTypeTest3x3(dimX1, dimX2)
	elseif wallTypeArg == "GlobalRef5x5r9"
		wallType = "GlobalRef5x5r9"
		println(wallType)
		C     = WallsTypeGlobalRef5x5r9(dimX1, dimX2)
	end

# 	display(C)
# 	println()
# 		exit(1)

	return C
end

# function ValidNeighborhood(dims::Array{Int,}, xi::Int, C::Array{Int, 2}) 
function ValidNeighborhood(dims::Tuple, xi::Int, C::Array{Int, 2}) 

	dimX1, dimX2, dimA, dimG, dimH = dims
	dimX = dimX1*dimX2

	Neighborhood_DEBUG = false
	dimX = dimX1*dimX2
	i, j = lin2cart(dimX1, dimX2, xi)
	actions  = Actions(dimA)
	nextState = zeros(Int, dimX, 1) # +1:stay at the same state, +1: num of poss next states
	numOfValidSteps = 0
	for neig = 1:dimA
		next_i, next_j = i+actions[neig, 1], j+actions[neig, 2]
		if ValidStep(dims, i, j, next_i, next_j, C, false)
			next_lin = cart2lin(dimX1, dimX2, next_i, next_j)
			nextState[next_lin] = next_lin 
			numOfValidSteps = numOfValidSteps + 1
		end
	end

	if Neighborhood_DEBUG
		println("+++++++++++++")
		println("start: Neighborhood_DEBUG")
		println("crntState=", lin2cart(dimX1, dimX2, xi))
		println("Successors#=", nextState[end])
		for i = 1:length(nextState)-1
			println(i, "=", lin2cart(dimX1, dimX2, nextState[i]))
		end
		println("end: Neighborhood_DEBUG")
		println("-------------")
	end

	return nextState
end
# function isValidNeighborhood(dims::Array{Int,}, C::Array{Int, 2}, next_i::Int, next_j::Int)
function isValidNeighborhood(dims::Tuple, C::Array{Int, 2}, next_i::Int, next_j::Int)

	dimX1, dimX2, dimA, dimG, dimH = dims
	dimX = dimX1*dimX2

	res = zeros(Int, dimX, 1)
	if ValidStep(dims, 0, 0, next_i, next_j, C, false)
		res[cart2lin(dimX1, dimX2, next_i, next_j)] = 1
	end
	return res
end

# function ValidStep(dims::Array{Int,}, crnt_i::Int, crnt_j::Int, next_i::Int, next_j::Int, C::Array{Int, 2}, debug::Bool=false)
function ValidStep(dims::Tuple, crnt_i::Int, crnt_j::Int, next_i::Int, next_j::Int, C::Array{Int, 2}, debug::Bool=false)
	dimX1, dimX2, dimA, dimG, dimH = dims

	dimX = dimX1*dimX2
	ValidStep_DEBUG = true
	res =  Bool(next_i>0) && Bool(next_j>0) && Bool(next_i<=dimX1) && Bool(next_j<=dimX2) && Bool(C[next_i, next_j])
	if debug
		# 		println("+++++++++++++")
		println("start: ValidStep_DEBUG crntStep=(", crnt_i, ",", crnt_j, ")")
		# 	nextLin = cart2lin(dimX1, dimX2, next_i, next_j)
		# 	nextCart = lin2cart(dimX1, dimX2, nextLin)
		if res 	
			println("VALID: nextStep=(", next_i, ",", next_j, ")")
		else
			println("INVALID: nextStep=(", next_i, ",", next_j, ")")
		end
		println("end: ValidStep_DEBUG")
		println("-------------")
	end
	return res
end

end

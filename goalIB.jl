function goalIB(dims, Pgxi_ai, Pg_idx::Int, Pxi, beta::Float64)

	dimG, dimH =  dims[4], dims[5]

	Pg      	= ones(dimG, 1, 1, 1, 1, 1)
# 	Pg      	= zeros(dimG, 1, 1, 1, 1, 1)
	Pg[Pg_idx, 1, 1, 1, 1, 1] 	= 1.0
	Pg = Normalize(Pg, (idxG, ))

	Pg_ai   	= pg_ai(Pgxi_ai, Pxi)

	# p(s, a, h, g) = p(a|s,g)p(g|h)p(h)p(s) 
	# p(a | s, h) = sum(g) p(a|s,g) p(g|h)

	Pg_h    	= Normalize(rand(dimG, dimH, 1, 1, 1, 1), (idxH, ))

	Ph      	= ph(Pg_h, Pg)
	Ph_g    	= ph_g(Pg_h, Pg, Ph) 

	Phxi_ai 	= phxi_ai(Pgxi_ai, Ph_g)
	Ph_ai   	= ph_ai(Phxi_ai, Pxi)

	Phxiai 		= Phxi_ai .* Ph .* Pxi
	# 	Ph              = sum(Phxiai, dims=(idxXi, idxAi))
	# 
# 	println("dddd")
# 	display(Ph[1, :, 1, 1, 1, 1]); println()
# 	display(Pg_h[:, :, 1, 1, 1, 1]); println()
# 
	for i in 1:3

		Pg_h 		= pg_h(Ph, Pxi, Pgxi_ai, Phxi_ai, Pg_ai, Ph_ai, beta) 

# 		display(Pg_h[:, :, 1, 1, 1, 1]); println()
# 		display(Ph[1, :, 1, 1, 1, 1]); println()
# 		display(reshape(Pxi[1, 1, :, 1, 1, 1], (dims[1],dims[2]))); println()
# 		display(reshape(Pg[:, 1, 1, 1, 1, 1], (dims[1],dims[2]))); println()
# 		display(Pg[:, 1, 1, 1, 1, 1]); println()

		Ph      	= ph(Pg_h, Pg)
		Ph_g    	= ph_g(Pg_h, Pg, Ph) 

		Phxi_ai 	= phxi_ai(Pgxi_ai, Ph_g)
		Ph_ai   	= ph_ai(Phxi_ai, Pxi)
		Phxiai 		= Phxi_ai .* Ph .* Pxi

		Pgh 	= pgh(Pg_h, Pg)
		i1 	= Igh(Pgh, Ph, Pg) 
		i2 	= Ih_xiai(Phxiai, Ph_ai, Phxi_ai) 

		println((i, i1, i2))
	end

	return Pg_h, Ph, Phxi_ai

end

macro dispMatrix(P)
	return :( display($P); println())
end

function main()

global P = randn(3, 3)
@dispMatrix(P)

end

main()

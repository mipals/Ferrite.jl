using Ferrite, MPI, HYPRE, Metis

# Initialize MPI and HYPRE
MPI.Init()
HYPRE.Init()

const comm = MPI.COMM_WORLD
const root = 0
const rank = MPI.Comm_rank(comm)
const comm_size = MPI.Comm_size(comm)

# No changes from serial solve
function assemble_element!(Ke::Matrix, fe::Vector, cellvalues::CellScalarValues)
    n_basefuncs = getnbasefunctions(cellvalues)
    fill!(Ke, 0)
    fill!(fe, 0)
    for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:n_basefuncs
            δu  = shape_value(cellvalues, q_point, i)
            ∇δu = shape_gradient(cellvalues, q_point, i)
            fe[i] += δu * dΩ
            for j in 1:n_basefuncs
                ∇u = shape_gradient(cellvalues, q_point, j)
                Ke[i, j] += (∇δu ⋅ ∇u) * dΩ
            end
        end
    end
    return Ke, fe
end

# No changes from serial solve other than looping over owned cells
function assemble_global(cellvalues::CellScalarValues, A::HYPREMatrix, b::HYPREVector, dh::DofHandler, ch::ConstraintHandler)
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    assembler = start_assemble(A, b)
    for cell in CellIterator(dh, getcellset(dh.grid, "##proc-$(rank)##"))
        reinit!(cellvalues, cell)
        assemble_element!(Ke, fe, cellvalues)
        apply_local!(Ke, fe, celldofs(cell), ch)
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    # TODO: Ferrite.finish_assemble maybe? Can be a no-op for SparseMatrixCSC
    HYPRE.finish_assemble!(assembler.A)
    return A, b
end

# Partition the grid using Metis.jl
function partition_grid!(grid)
    # TODO: Can this be done on all ranks? Not sure if Metis is deterministic.
    if rank == root
        if comm_size == 1
            parts = ones(Cint, getncells(grid))
        else
            cell_connectivity = Ferrite.create_incidence_matrix(grid)
            parts = Metis.partition(cell_connectivity, comm_size)
        end
    else
        parts = Vector{Cint}(undef, getncells(grid))
    end
    MPI.Bcast!(parts, comm)

    # Create the cell sets based on the Metis partition
    sets = [Set{Int}() for _ in 1:comm_size]
    for (cell_id, part_id) in pairs(parts)
        push!(sets[part_id], cell_id)
    end
    for p in 0:comm_size-1
        addcellset!(grid, "##proc-$p##", sets[p+1])
    end
    return grid
end

function main()

    # Create the grid
    grid = generate_grid(Quadrilateral, (20, 20));

    # Partition the mesh
    partition_grid!(grid)

    # Create the DofHandler
    dh = DofHandler(grid)
    push!(dh, :u, 1)
    close!(dh);

    # Renumber by part
    all = Set{Int}()
    sets = [Set{Int}() for _ in 1:comm_size]
    cc = CellCache(dh)
    for p in 0:comm_size-1
        set = sets[p+1]
        for cell_id in getcellset(grid, "##proc-$p##")
            reinit!(cc, cell_id)
            union!(set, cc.dofs)
        end
        setdiff!(set, all)
        union!(all, set)
    end
    iperm = Int[]
    rank_dof_ranges = UnitRange{Int}[]
    for set in sets
        push!(rank_dof_ranges, (length(iperm)+1):(length(iperm)+length(set)))
        append!(iperm, sort!(collect(set)))
    end
    perm = invperm(iperm)
    renumber!(dh, perm)
    rank_dof_range = rank_dof_ranges[rank+1]

    # FE Values
    cellvalues = let
        dim = 2
        ip = Lagrange{dim, RefCube, 1}()
        qr = QuadratureRule{dim, RefCube}(2)
        CellScalarValues(qr, ip)
    end

    # ConstraintHandler
    ch = ConstraintHandler(dh)
    ∂Ω = union(
        getfaceset(grid, "left"),
        getfaceset(grid, "right"),
        getfaceset(grid, "top"),
        getfaceset(grid, "bottom"),
    )
    dbc = Dirichlet(:u, ∂Ω, (x, t) -> 0)
    add!(ch, dbc);
    close!(ch)


    # Set up HYPRE arrays
    ilower, iupper = extrema(rank_dof_range)
    A = HYPREMatrix(comm, ilower, iupper)
    b = HYPREVector(comm, ilower, iupper)

    # Assemble
    assemble_global(cellvalues, A, b, dh, ch)

    # Set up solver and solve
    precond = HYPRE.BoomerAMG()
    solver = HYPRE.PCG(; Precond = precond)
    xh = HYPRE.solve(solver, A, b)

    # Copy solution from HYPRE to Julia
    x = Vector{Float64}(undef, length(rank_dof_range))
    copy!(x, xh)

    # Collect to root rank
    if rank == root
        X = Vector{Float64}(undef, ndofs(dh))
        counts = length.(rank_dof_ranges)
        MPI.Gatherv!(x, VBuffer(X, counts), comm)
    else
        MPI.Gatherv!(x, nothing, comm)
    end

    ### Exporting to VTK
    if rank == root
        vtk_grid("heat_equation", dh) do vtk
            vtk_point_data(vtk, dh, X)
        end
        @show norm(X)
        @show norm(X) ≈ 3.307743912641305
    end
end

# Run it!
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

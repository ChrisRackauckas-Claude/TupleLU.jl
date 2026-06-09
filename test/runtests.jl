const GROUP = get(ENV, "GROUP", "All")

if GROUP == "QA"
    using Pkg
    Pkg.activate(joinpath(@__DIR__, "qa"))
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
    include(joinpath(@__DIR__, "qa", "qa.jl"))
else
    using ReTestItems
    using TupleLU

    runtests(TupleLU; nworkers = 2)
end

using SciMLTesting, TupleLU, JET, Test

run_qa(
    TupleLU;
    explicit_imports = true,
    aqua_broken = (:unbound_args,),  # (TupleMatrix{M,N})(x::NTuple{L,T}) trips test_unbound_args — https://github.com/SciML/TupleLU.jl/issues/8
    jet_broken = true,               # similar_type undefined in getproperty + degenerate TupleMatrix __lu ctors — https://github.com/SciML/TupleLU.jl/issues/8
    jet_kwargs = (; target_defined_modules = true),
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            # @propagate_inbounds/setindex: Base internals; HermOrSym: LinearAlgebra internal
            ignore = (Symbol("@propagate_inbounds"), :setindex, :HermOrSym),
        ),
    ),
)

@testset "public API appears in README" begin
    readme = read(joinpath(pkgdir(TupleLU), "README.md"), String)
    missing = filter(name -> !occursin(String(name), readme), public_api_names(TupleLU))
    @test isempty(missing)
end

using SciMLTesting, TupleLU, JET

run_qa(
    TupleLU;
    explicit_imports = true,
    api_docs_kwargs = (; rendered = true),
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

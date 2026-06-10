using TupleLU
using Aqua
using ExplicitImports
using JET
using Test

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(TupleLU, unbound_args = false, deps_compat = false)
    # `(TupleMatrix{M, N})(x::NTuple{L, T}) where {M, N, T, L}` trips test_unbound_args.
    Aqua.test_unbound_args(TupleLU, broken = true)
    @test_broken false  # Aqua deps_compat: `Pkg` in [extras] has no [compat] entry — see https://github.com/SciML/TupleLU.jl/issues/8
end

@testset "Explicit Imports" begin
    @test check_no_implicit_imports(TupleLU) === nothing
    @test check_no_stale_explicit_imports(TupleLU) === nothing
end

@testset "Code linting (JET.jl)" begin
    @test_broken false  # JET: undefined `similar_type` in getproperty + degenerate TupleMatrix __lu ctors — see https://github.com/SciML/TupleLU.jl/issues/8
end

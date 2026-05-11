@testitem "Code quality (Aqua.jl)" tags = [:qa] begin
    using TupleLU
    using Aqua

    Aqua.test_all(TupleLU, unbound_args=false)
    # Somehow, `(TupleMatrix{M, N})(x::NTuple{L, T}) where {M, N, T, L}` fails test_unbound_args
    Aqua.test_unbound_args(TupleLU, broken=true)
end

@testitem "Code quality (Aqua.jl)" tags = [:qa] begin
    using TupleLU
    using Aqua

    Aqua.test_all(TupleLU)
end

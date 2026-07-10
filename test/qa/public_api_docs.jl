using TupleLU
using Test

@testset "public API documentation" begin
    public_names = filter(!=(:TupleLU), names(TupleLU; all = false, imported = false))
    @test Set(public_names) == Set([:LU, :TupleMatrix, :issuccess, :lu])

    for name in public_names
        binding = Docs.Binding(TupleLU, name)
        @test Docs.hasdoc(binding)
    end

    readme = read(joinpath(pkgdir(TupleLU), "README.md"), String)
    for name in public_names
        @test occursin(string(name), readme)
    end
end

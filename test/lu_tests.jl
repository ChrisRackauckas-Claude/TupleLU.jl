@testitem "LU" begin
using TupleLU
using LinearAlgebra
using Test

@testset "LU utils" begin
    # column-major: [1.0 2.0; 3.0 4.0]
    F = lu(TupleMatrix{2,2}((1.0, 3.0, 2.0, 4.0)))

    @test @inferred((F -> F.p)(F)) === (2, 1)
    @test occursin(r"L factor.*U factor"s, sprint(show, MIME("text/plain"), F))
end

@testset "TupleMatrix construction" begin
    A = TupleMatrix{2,3}((1.0, 4.0, 2.0, 5.0, 3.0, 6.0))   # [1 2 3; 4 5 6]
    @test size(A) == (2, 3)
    @test A[1, 1] == 1.0 && A[2, 3] == 6.0
    @test A == TupleMatrix{2,3}([1.0 2.0 3.0; 4.0 5.0 6.0])
    @test_throws ArgumentError TupleMatrix{2,2,Float64,5}((1.0, 2.0, 3.0, 4.0, 5.0))
    @test_throws DimensionMismatch TupleMatrix{2,3}([1.0 2.0; 3.0 4.0])
end

@testset "LU decomposition ($m×$n, pivot=$pivot, wrapper=$wrap)" for
        pivot in (true, false), m in 1:4, n in 1:4, wrap in (identity, Symmetric, Hermitian)

    a = if m == n
        M = Float64[i == j ? i + 1 : i + j for i in 1:m, j in 1:n]
        wrap(TupleMatrix{m,n}(M))
    elseif wrap !== identity
        continue
    else
        TupleMatrix{m,n}(Float64[(i - 1) * n + j for i in 1:m, j in 1:n])
    end

    L, U, p = lu(a, Val(pivot); check=false)

    if m == n
        @test L isa LowerTriangular{Float64,<:TupleMatrix{m,m,Float64}}
        @test U isa UpperTriangular{Float64,<:TupleMatrix{m,m,Float64}}
    else
        @test L isa TupleMatrix{m,min(m, n),Float64}
        @test U isa TupleMatrix{min(m, n),n,Float64}
    end
    @test p isa NTuple{m,Int}

    A_mat = Matrix(a)
    L_mat = Matrix(L)
    U_mat = Matrix(U)
    p_vec = collect(p)

    if pivot
        @test sort(p_vec) == collect(1:m)
    else
        @test p_vec == collect(1:m)
    end

    # L is unit lower triangular
    for i in 1:m, j in (i + 1):min(m, n)
        @test iszero(L_mat[i, j])
    end
    for i in 1:min(m, n)
        @test L_mat[i, i] == 1
    end

    # U is upper triangular
    for i in 1:min(m, n), j in 1:(i - 1)
        @test iszero(U_mat[i, j])
    end

    # decomposition is correct
    @test L_mat * U_mat ≈ A_mat[p_vec, :]
end

@testset "LU division ($m×$n)" for m in 1:4, n in 1:4
    A = TupleMatrix{m,m}(Float64[i == j ? i + 1 : 0.5 * (i + j) for i in 1:m, j in 1:m])
    F = lu(A)
    A_mat = Matrix(A)
    b_col = rand(m, n)
    b_line = rand(n, m)

    @test A_mat \ b_col ≈ F \ b_col
    @test A_mat \ b_col[:, 1] ≈ F \ b_col[:, 1]
    @test b_line / A_mat ≈ b_line / F
end

@testset "PivotingStrategy M=N=$m" for m in (2, 3, 4)
    A = TupleMatrix{m,m}(Float64[i == j ? i + 1 : 0.5 * (i + j) for i in 1:m, j in 1:m])
    @test lu(A, Val(false)).p == lu(A, NoPivot()).p
    @test lu(A, Val(true)).p == lu(A, RowMaximum()).p
end

@testset "LU singularity check M=N=$m" for m in (2, 3, 4)
    A = TupleMatrix{m,m}(ones(m, m))
    @test_throws SingularException lu(A)
    @test !issuccess(lu(A; check=false))
end

@testset "LU integer input promotes to Float64" begin
    L, U, p = lu(TupleMatrix{2,2}((1, 3, 2, 4)))
    @test eltype(L) === Float64
    @test eltype(U) === Float64
end

end # @testitem "LU"

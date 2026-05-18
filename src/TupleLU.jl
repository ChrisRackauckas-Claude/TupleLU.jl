"""
    TupleLU

LU factorization for small statically-sized matrices backed by an `NTuple`.

The implementation is adapted from StaticArrays.jl (MIT license),
[`src/lu.jl`](https://github.com/JuliaArrays/StaticArrays.jl/blob/f02280c5c1f05c56a52fbe45168e2961416c3f89/src/lu.jl),
specialized for an `NTuple`-backed matrix type. Encoding the size in the type lets the
compiler unroll the factorization for sizes up to roughly 14×14; larger inputs fall back
to `LinearAlgebra.lu(::Matrix, …)`.
"""
module TupleLU

using LinearAlgebra: LinearAlgebra, Hermitian, LowerTriangular, NoPivot, RowMaximum
using LinearAlgebra: SingularException, Symmetric, UnitLowerTriangular, UnitUpperTriangular
using LinearAlgebra: UpperTriangular, issuccess, lu

export TupleMatrix, LU, lu, issuccess

"""
    TupleMatrix{M, N, T, L} <: AbstractMatrix{T}

Statically-sized `M × N` matrix backed by an `NTuple{L, T}`, where `L = M * N`.

Storage is column-major (matching `Matrix`). Encoding the dimensions in the type lets
factorizations like [`lu`](@ref) be fully unrolled by the compiler.

# Constructors

    TupleMatrix{M, N}(data::NTuple{L, T})
    TupleMatrix{M, N, T}(data::NTuple{L, T})
    TupleMatrix{M, N}(data::AbstractMatrix{T})
    TupleMatrix{M, N, T}(data::AbstractMatrix{T})
    TupleMatrix{M, N, T}(f)

The first two forms wrap an existing column-major tuple, checking that `M * N == L`. The
two `AbstractMatrix` forms copy entries from a runtime-sized matrix after verifying its
shape matches `(M, N)`. The final form constructs entries on the fly by calling `f(i, j)`
for each row/column index pair — this is the workhorse the factorization uses to assemble
each new submatrix.
"""
struct TupleMatrix{M, N, T, L} <: AbstractMatrix{T}
    data::NTuple{L, T}
    function TupleMatrix{M, N, T, L}(x::NTuple{L, T}) where {M, N, T, L}
        M * N == L || throw(ArgumentError("Length of the input tuple ($(L)) does not match \
            the size of the matrix ($(M)×$(N) = $(M * N))"))
        return new{M, N, T, L}(x)
    end
end
TupleMatrix{M, N}(x::NTuple{L, T}) where {M, N, T, L} = TupleMatrix{M, N, T, L}(x)
TupleMatrix{M, N, T}(x::NTuple{L, T}) where {M, N, T, L} = TupleMatrix{M, N, T, L}(x)
TupleMatrix{M, N}(A::AbstractMatrix{T}) where {M,N,T} = TupleMatrix{M, N, T}(A)
function TupleMatrix{M, N, T}(A::AbstractMatrix{T}) where {M, N, T}
    size(A) == (M, N) || throw(DimensionMismatch("expected size ($M, $N), got $(size(A))"))
    @inbounds TupleMatrix{M, N, T, M * N}(ntuple(idx -> T(A[idx]), Val(M * N)))
end

@generated function TupleMatrix{M, N, T}(f::F) where {M, N, T, F}
    tup = Expr(:tuple)
    for j in 1:N, i in 1:M
        push!(tup.args, :(f($i, $j)::T))
    end
    return quote
        $(Expr(:meta, :inline))
        TupleMatrix{M, N, T, $(M * N)}($tup)
    end
end

Base.size(::TupleMatrix{M, N}) where {M, N} = (M, N)
Base.length(@nospecialize(t::TupleMatrix)) = nfields(t.data)

_throw_boundserror(A, I) = (@noinline; throw(BoundsError(A, I)))

Base.@propagate_inbounds Base.getindex(t::TupleMatrix, i) = t.data[i]
Base.@propagate_inbounds function Base.getindex(
        t::TupleMatrix{M}, i::Integer, j::Integer) where {M}
    return t.data[i + (j - 1) * M]
end
# Row slice: returns the i-th row as a `NTuple{N, T}`. The factorization uses this to
# capture `A[kp, :]` (the pivot row) without going through an intermediate matrix.
Base.@propagate_inbounds function Base.getindex(
        t::TupleMatrix{M, N, T, L}, i, c::Colon) where {M, N, T, L}
    @boundscheck 1 ≤ i ≤ M || _throw_boundserror(t, (i, c))
    return @inbounds ntuple(idx -> t.data[i + (idx - 1) * M], Val(N))
end
Base.IndexStyle(::Type{<:TupleMatrix}) = IndexLinear()

const TupleLUMatrix{N, M, T} = Union{
    TupleMatrix{N, M, T},
    Symmetric{T, <:TupleMatrix{N, M, T}},
    Hermitian{T, <:TupleMatrix{N, M, T}},
}
const TupleULT{TA} = Union{
    UpperTriangular{TA,<:TupleMatrix},
    LowerTriangular{TA,<:TupleMatrix},
    UnitUpperTriangular{TA,<:TupleMatrix},
    UnitLowerTriangular{TA,<:TupleMatrix},
}

const _PIVOT_OPTIONS = (:(Val{true}), :(Val{false}), :NoPivot, :RowMaximum)

# Element type closed under the arithmetic used by Gaussian elimination. Used to pick the
# eltype of L and U up front, since the factorization needs to divide entries of `T`.
arithmetic_closure(::Type{T}) where {T} = typeof((one(T) * zero(T) + zero(T)) / one(T))

"""
    LU{L, U, p}

LU factorization returned by [`lu`](@ref) on a [`TupleMatrix`](@ref).

This mirrors `LinearAlgebra.LU` but stores the row-permutation `p` as an `NTuple` rather
than a `Vector`, keeping the factorization fully type-stable. The components destructure
in the usual order: `L, U, p = F`.
"""
struct LU{L, U, p}
    L::L
    U::U
    p::p
end

# Iteration interface so that `L, U, p = F` destructures an `LU`.
Base.iterate(S::LU) = (S.L, Val(:U))
Base.iterate(S::LU, ::Val{:U}) = (S.U, Val(:p))
Base.iterate(S::LU, ::Val{:p}) = (S.p, Val(:done))
Base.iterate(S::LU, ::Val{:done}) = nothing

@inline function Base.getproperty(F::LU, s::Symbol)
    if s === :P
        U = getfield(F, :U)
        p = getfield(F, :p)
        return one(similar_type(p, Size(U)))[:, invperm(p)]
    else
        return getfield(F, s)
    end
end

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, F::LU)
    println(io, LU)  # Avoid the parametric type — the factors below are more informative.
    println(io, "L factor:")
    show(io, mime, F.L)
    println(io, "\nU factor:")
    show(io, mime, F.U)
end


"""
    LinearAlgebra.lu(A::TupleMatrix; check=true)
    LinearAlgebra.lu(A::TupleMatrix, pivot; check=true)

LU factorization of a [`TupleMatrix`](@ref). Returns a [`TupleLU.LU`](@ref).

`pivot` selects partial pivoting (`Val(true)` / `RowMaximum()`, the default) or no
pivoting (`Val(false)` / `NoPivot()`). When `check=true`, a `SingularException` is thrown
if the upper factor has a zero on its diagonal.

For square inputs the factors are returned wrapped in `LowerTriangular` and
`UpperTriangular`. Rectangular inputs return raw [`TupleMatrix`](@ref) factors because
`Base` does not support triangular wrappers around rectangular matrices.
"""
LinearAlgebra.lu(A::TupleLUMatrix; check = true) = lu(A, Val(true); check = check)

for pv in _PIVOT_OPTIONS
    # Define each pivot method individually to avoid ambiguities between the
    # `Val{true}/Val{false}` flag and `NoPivot`/`RowMaximum` instances.
    @eval function LinearAlgebra.lu(A::TupleLUMatrix, pivot::$pv; check = true)
        L, U, p = _lu(A, pivot, check)
        return LU(L, U, p)
    end

    @eval function LinearAlgebra.lu(
            A::TupleLUMatrix{N, N}, pivot::$pv; check = true) where {N}
        L, U, p = _lu(A, pivot, check)
        return LU(LowerTriangular(L), UpperTriangular(U), p)
    end
end

# Index of the first zero on the diagonal, or 0 if none. Used to detect singularity.
_first_zero_on_diagonal(A::TupleULT) = _first_zero_on_diagonal(A.data)
function _first_zero_on_diagonal(A::TupleLUMatrix{M, N, T}) where {M, N, T}
    if @generated
        quote
            $(map(i -> :(A[$i, $i] == zero(T) && return $i), 1:min(M, N))...)
            0
        end
    else
        for i in 1:min(M, N)
            A[i, i] == 0 && return i
        end
        return 0
    end
end

LinearAlgebra.issuccess(F::LU) = _first_zero_on_diagonal(F.U) == 0

# Above this many entries the unrolled implementation pushes type inference hard enough
# that compile time dominates; we delegate to `Base.lu(::Matrix, …)` instead.
const UNROLL_LIMIT = 14 * 14

@generated function _lu(A::TupleLUMatrix{M, N, T}, pivot, check) where {M, N, T}
    if M * N ≤ UNROLL_LIMIT
        _pivot = if isdefined(LinearAlgebra, :PivotingStrategy)
            pivot === RowMaximum ? Val(true) :
            pivot === NoPivot    ? Val(false) :
            pivot()
        else
            pivot()
        end
        quote
            L, U, P = __lu(A, $(_pivot))
            if check
                i = _first_zero_on_diagonal(U)
                i == 0 || throw(SingularException(i))
            end
            return L, U, P
        end
    else
        _pivot = if isdefined(LinearAlgebra, :PivotingStrategy)
            pivot === Val{true}  ? RowMaximum() :
            pivot === Val{false} ? NoPivot() :
            pivot()
        else
            pivot()
        end
        # `f.L`'s eltype is not type-inferable, so derive the eltype up front from
        # `arithmetic_closure(T)` rather than reading it back off the result.
        T2 = arithmetic_closure(T)
        K = min(M, N)
        quote
            # Delegate to Base for large matrices to avoid runaway compile times.
            f = lu(Matrix(A), $(_pivot); check = check)
            L = TupleMatrix{$M, $K, $T2}(Matrix{$T2}(f.L))
            U = TupleMatrix{$K, $N, $T2}(Matrix{$T2}(f.U))
            p = NTuple{$M, Int}(f.p)
            return L, U, p
        end
    end
end

# --- Recursive (bordered) factorization --------------------------------------------------
#
# `__lu` performs a single elimination step: factor the first column, then recurse on the
# Schur complement `A[ps, 2:N] - Ls * Ufirst[2:N]`. The base cases below cover the
# degenerate shapes (zero rows/cols, 1×1, 1×N, M×1) where there is no recursion to do.

__lu(A::TupleMatrix{0, 0, T}, ::Val{Pivot}) where {T, Pivot} =
    (TupleMatrix{0, 0, typeof(one(T)), 0}(), A, NTuple{0, Int}())

__lu(A::TupleMatrix{0, 1, T}, ::Val{Pivot}) where {T, Pivot} =
    (TupleMatrix{0, 0, typeof(one(T)), 0}(), A, NTuple{0, Int}())

__lu(A::TupleMatrix{0, N, T}, ::Val{Pivot}) where {T, N, Pivot} =
    (TupleMatrix{0, 0, typeof(one(T)), 0}(), A, NTuple{0, Int}())

__lu(A::TupleMatrix{1, 0, T}, ::Val{Pivot}) where {T, Pivot} =
    (TupleMatrix{1, 0, typeof(one(T)), 0}(), TupleMatrix{0, 0, T, 0}(), NTuple{1, Int}(1))

__lu(A::TupleMatrix{M, 0, T}, ::Val{Pivot}) where {T, M, Pivot} =
    (TupleMatrix{M, 0, typeof(one(T)), 0}(), TupleMatrix{0, 0, T, 0}(),
        NTuple{M, Int}(1:M))

__lu(A::TupleMatrix{1, 1, T}, ::Val{Pivot}) where {T, Pivot} =
    (TupleMatrix{1, 1, T, 1}((one(T),)), A, NTuple{1, Int}(1))

__lu(A::LinearAlgebra.HermOrSym{T, <:TupleMatrix{1, 1, T}}, ::Val{Pivot}) where {T, Pivot} =
    (TupleMatrix{1, 1, T, 1}((one(T),)), A.data, NTuple{1, Int}(1))

__lu(A::TupleMatrix{1, N, T}, ::Val{Pivot}) where {N, T, Pivot} =
    (TupleMatrix{1, 1, T, 1}((one(T),)), A, NTuple{1, Int}(1))

# Single-column elimination step: pick the pivot, scale the rest of the column by its
# inverse, and emit a 1×1 U.
function __lu(A::TupleMatrix{M, 1, T}, ::Val{Pivot}) where {M, T, Pivot}
    P = promote_type(T, typeof(inv(one(T))))
    @inbounds begin
        kp = 1
        if Pivot
            amax = abs(A[1, 1])
            for i in 2:M
                absi = abs(A[i, 1])
                if absi > amax
                    kp = i
                    amax = absi
                end
            end
        end
        ps = tailindices(Val{M})
        if kp != 1
            ps = Base.setindex(ps, 1, kp - 1)
        end
        # Scale the first column by the inverse of the pivot.
        Akk = A[kp, 1]
        Akkinv = inv(Akk)
        Ltup = let A = A.data, ps = ps, Akkinv = Akkinv
            if isfinite(Akkinv)
                ntuple(Val(M)) do i
                    i == 1 ? one(P) : (A[ps[i - 1]] * Akkinv)::P
                end
            else
                # Pivot is zero/Inf; emit zero multipliers so singularity is reported
                # downstream by `_first_zero_on_diagonal`.
                ntuple(Val(M)) do i
                    i == 1 ? one(P) : zero(P)
                end
            end
        end
        L = TupleMatrix{M, 1, P}(Ltup)
        U = TupleMatrix{1, 1, P}((P(Akk),))
        p = (kp, ps...)
    end
    return (L, U, p)
end

# General `M × N` step: factor the first column, recurse on the Schur complement, and
# stitch the results together. Once `M` and `N` are known to the compiler this expands
# into a fully unrolled factorization.
function __lu(A::TupleLUMatrix{M, N, T}, ::Val{Pivot}) where {M, N, T, Pivot}
    P = promote_type(T, typeof(inv(one(T))))
    @inbounds begin
        kp = 1
        if Pivot
            amax = abs(A[1, 1])
            for i in 2:M
                absi = abs(A[i, 1])
                if absi > amax
                    kp = i
                    amax = absi
                end
            end
        end
        ps = tailindices(Val{M})::NTuple{M - 1, Int}
        if kp != 1
            ps = Base.setindex(ps, 1, kp - 1)
        end

        Ufirst = A[kp, :]
        # Scale the first column by the inverse of the pivot.
        Akk = A[kp, 1]
        Akkinv = inv(Akk)
        Ls = let A = A.data, ps = ps, Akkinv = Akkinv
            if isfinite(Akkinv)
                ntuple(i -> (A[ps[i]] * Akkinv)::P, Val(M - 1))
            else
                ntuple(_ -> zero(P), Val(M - 1))
            end
        end

        # Schur complement: A[ps, 2:N] - Ls * Ufirst[2:N].
        Arest = let A = A, Ufirst = Ufirst, Ls = Ls, ps = ps
            TupleMatrix{M - 1, N - 1, P}() do i, j
                A[ps[i], j + 1] - Ls[i] * Ufirst[j + 1]
            end
        end
        Lrest, Urest, prest = __lu(Arest, Val(Pivot))
        p = let ps = ps, prest = prest
            (kp, ntuple(i -> ps[prest[i]], Val(M - 1))...)
        end

        # L is M × min(M, N): unit-augmented `Ls` over `Lrest`. The recursive permutation
        # `prest` is applied to `Ls` so the first column lines up with U.
        K = min(M, N)
        # TODO: is the let block necessary?
        L = let Ls = Ls, Lrest = Lrest, prest = prest
            TupleMatrix{M, K, P}() do i, j
                if j == 1
                    i == 1 ? one(P) : Ls[prest[i - 1]]
                else
                    i == 1 ? zero(P) : Lrest[i - 1, j - 1]
                end
            end
        end
        # U is min(M, N) × N: `Ufirst` on top, `Urest` below.
        # TODO: is the let block necessary?
        U = let Ufirst = Ufirst, Urest = Urest
            TupleMatrix{K, N, P}() do i, j
                if i == 1
                    P(Ufirst[j])
                else
                    j == 1 ? zero(P) : P(Urest[i - 1, j - 1])
                end
            end
        end
    end  # @inbounds
    return (L, U, p)
end

@inline tailindices(::Type{Val{M}}) where {M} = ntuple(i -> i + 1, Val(M - 1))

# TODO: StaticArrays avoids allocations here by promoting `v[F.p]` to an `SVector` and
# dispatching to a specialized generated `ldiv!`:
# https://github.com/JuliaArrays/StaticArrays.jl/blob/f02280c5c1f05c56a52fbe45168e2961416c3f89/src/triangular.jl#L15
function Base.:(\)(F::LU, v::AbstractVector)
    length(F.p) != length(v) &&
        @noinline throw(DimensionMismatch("arguments must have the same number of rows"))
    return @inbounds F.U \ (F.L \ [v[i] for i in F.p])
end

# Both StaticArrays and us allocate the permuted B matrix
function Base.:(\)(F::LU, B::AbstractMatrix)
    length(F.p) != size(B, 1) &&
        @noinline throw(DimensionMismatch("arguments must have the same number of rows"))
    P = similar(B)
    @inbounds for j in 1:size(B, 2)
        for (i, p) in enumerate(F.p)
            P[i, j] = B[p, j]
        end
    end
    return F.U \ (F.L \ P)
end

# TODO: Get rid of the collect
Base.:(/)(B::AbstractMatrix, F::LU) = @inbounds ((B / F.U) / F.L)[:, collect(invperm(F.p))]

end # module TupleLU

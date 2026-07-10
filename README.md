# TupleLU

[![Build Status](https://github.com/Drvi/TupleLU.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Drvi/TupleLU.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

`TupleLU.jl` provides LU factorization for small, statically-sized matrices backed by
tuples. The exported API is:

- `TupleMatrix`: an `NTuple`-backed `AbstractMatrix` with dimensions encoded in the type.
- `LU`: the tuple-backed LU factorization object returned by `lu(::TupleMatrix)`.
- `lu`: the `LinearAlgebra.lu` methods for `TupleMatrix`, `Symmetric{<:TupleMatrix}`,
  and `Hermitian{<:TupleMatrix}` inputs.
- `issuccess`: the `LinearAlgebra.issuccess` method for `TupleLU.LU` factorizations.

```julia
using LinearAlgebra
using TupleLU

A = TupleMatrix{2, 2}((1.0, 3.0, 2.0, 4.0))
F = lu(A)
L, U, p = F
issuccess(F)
```

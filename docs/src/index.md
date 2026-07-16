```@meta
CurrentModule = TupleLU
```

# TupleLU.jl

TupleLU provides LU factorization for small, statically-sized matrices backed by tuples.

## API

```@docs
TupleLU
TupleMatrix
LU
LinearAlgebra.lu(::TupleLU.TupleLUMatrix)
LinearAlgebra.issuccess(::LU)
```

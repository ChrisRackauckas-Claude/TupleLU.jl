using TupleLU, BenchmarkTools
using TupleLU: TupleMatrix, issuccess
import TupleLU as TLU

const SUITE = BenchmarkGroup()

# Small dense systems as TupleMatrix
m2 = TupleMatrix{2, 2}((1.0, 3.0, 2.0, 4.0))
m3 = TupleMatrix{3, 3}((2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0))
m4 = TupleMatrix{4, 4}(
    (4.0, 1.0, 0.5, 0.0, 1.0, 3.0, 0.2, 0.1, 0.5, 0.2, 5.0, 0.3, 0.0, 0.1, 0.3, 2.0)
)

b2 = [1.0, 2.0]
b3 = [1.0, 0.0, 1.0]
b4 = [1.0, 2.0, 3.0, 4.0]

# =============================================================================
# LU factorization
# =============================================================================

SUITE["lu"] = BenchmarkGroup()

SUITE["lu"]["2x2"] = @benchmarkable TLU.lu($m2)
SUITE["lu"]["3x3"] = @benchmarkable TLU.lu($m3)
SUITE["lu"]["4x4"] = @benchmarkable TLU.lu($m4)

# =============================================================================
# Solve through factorization
# =============================================================================

SUITE["solve"] = BenchmarkGroup()

F3 = TLU.lu(m3)
F4 = TLU.lu(m4)
SUITE["solve"]["2x2"] = @benchmarkable $(TLU.lu(m2)) \ $b2
SUITE["solve"]["3x3"] = @benchmarkable $F3 \ $b3
SUITE["solve"]["4x4"] = @benchmarkable $F4 \ $b4
SUITE["solve"]["issuccess"] = @benchmarkable issuccess($F4)

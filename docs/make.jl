using Documenter
using LinearAlgebra
using TupleLU

makedocs(;
    modules = [TupleLU],
    sitename = "TupleLU.jl",
    pages = [
        "Home" => "index.md",
    ],
    checkdocs = :exports,
)

deploydocs(; repo = "github.com/SciML/TupleLU.jl.git", push_preview = true)

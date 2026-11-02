using Documenter, DocumenterVitepress, PowerScaling

makedocs(
    sitename="PowerScaling.jl",
    modules=[PowerScaling],
    format=DocumenterVitepress.MarkdownVitepress(
        repo="github.com/nsiccha/PowerScaling.jl", devurl="dev", devbranch="master"),
    pages=["Home" => "index.md", "API" => "api.md"],
    checkdocs=:exports,
)

let redirect = joinpath(@__DIR__, "build", "index.html")
    isfile(redirect) || write(redirect, """
    <!DOCTYPE html><html><head><meta http-equiv="refresh" content="0; url=dev/"></head>
    <body>Redirecting to <a href="dev/">dev</a>...</body></html>
    """)
end

DocumenterVitepress.deploydocs(
    repo="github.com/nsiccha/PowerScaling.jl", devbranch="master", push_preview=true)

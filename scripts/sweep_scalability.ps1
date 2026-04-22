$threads = @(1, 2, 4, 8)

# 2. Loop through each thread count
foreach ($t in $threads) {
    Write-Host "---------------------------------------" -ForegroundColor Cyan
    Write-Host "EXECUTING: Threads = $t"
    Write-Host "---------------------------------------" -ForegroundColor Cyan

    cargo run --profile profiling -- --local $t
}

Write-Host "Sweep finished. Updating scalability graphs..." -ForegroundColor Green
py scripts/scalability_graph.py
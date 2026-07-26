param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PipelineArguments
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtual environment not found. Follow the setup steps in README.md."
}

Push-Location $ProjectRoot
try {
    & $Python -m src.pipeline @PipelineArguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}

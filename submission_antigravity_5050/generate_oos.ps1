<#
.SYNOPSIS
    1-Click Lynx 3-File Submission Generator for Team BOMBAGOAT.

.DESCRIPTION
    Runs the BOMBAGOAT 50/50 ensemble on 'insample', 'outofsample1', and 'outofsample2'
    and generates the exact 3 files required by the Lynx portal:
      <team>_insample.csv
      <team>_outofsample1.csv
      <team>_outofsample2.csv

.EXAMPLE
    .\generate_oos.ps1 -Team team06
    .\generate_oos.ps1 -Team team-06 -DataDir "C:\path\to\downloaded_oos"
#>
param(
    [string]$Team = "team06",
    [string]$DataDir = "$PSScriptRoot\..\data"
)

$ErrorActionPreference = "Stop"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  TEAM BOMBAGOAT: LYNX 3-FILE SUBMISSION PIPELINE" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$Script = Resolve-Path "$PSScriptRoot\generate_lynx_submission.py"

python "$Script" --team "$Team" --data-dir "$DataDir" --out-dir "$PSScriptRoot"

Write-Host "`nSubmission files generated in: $PSScriptRoot" -ForegroundColor Green
Get-ChildItem -Path "$PSScriptRoot" -Filter "*$Team*.csv" | ForEach-Object {
    Write-Host "  --> $($_.Name) ($($_.Length) bytes)" -ForegroundColor Yellow
}

$InsamplePath = (Resolve-Path "$PSScriptRoot\${Team}_insample.csv").Path
Set-Clipboard -Value $InsamplePath
Write-Host "`nPath to ${Team}_insample.csv copied to Windows Clipboard!" -ForegroundColor Magenta

param(
    [Parameter(Mandatory = $true)][string]$AssetDirectory,
    [string]$Tag = 'weights-v1'
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$manifest = Get-Content -LiteralPath (Join-Path $repoRoot 'code/pretrained_assets.json') -Raw | ConvertFrom-Json
if ($Tag -ne $manifest.release_tag) { throw 'Tag must match the public checkpoint manifest.' }
$assetRoot = (Resolve-Path -LiteralPath $AssetDirectory).Path
$assetPaths = @()
foreach ($asset in $manifest.assets) {
    $path = Join-Path $assetRoot $asset.name
    $file = Get-Item -LiteralPath $path
    if ($file.Length -ne $asset.bytes) { throw "Size mismatch: $($asset.name)" }
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $asset.sha256) {
        throw "Checksum mismatch: $($asset.name)"
    }
    $assetPaths += $path
}
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { throw 'Install GitHub CLI and run gh auth login first.' }
$login = & gh api user --jq .login
if ($LASTEXITCODE -ne 0 -or $login.Trim() -ne 'jackzhangqing') { throw 'Authenticate GitHub CLI as jackzhangqing first.' }
$releaseArgs = @('release', 'create', $Tag, '--repo', 'jackzhangqing/cigAMT', '--target', 'main',
    '--title', 'cigAMT pretrained baselines', '--notes-file', (Join-Path $repoRoot 'code/WEIGHTS_RELEASE_NOTES.md')) + $assetPaths
& gh @releaseArgs
if ($LASTEXITCODE -ne 0) { throw 'Release publication did not complete. Inspect the Releases page before retrying; do not overwrite assets blindly.' }

param(
    [Parameter(Mandatory = $true)][string]$AssetDirectory,
    [string]$Tag = 'weights-v1'
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$manifest = Get-Content -LiteralPath (Join-Path $repoRoot 'code/pretrained_assets.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($Tag -ne $manifest.release_tag) { throw 'Tag must match the public checkpoint manifest.' }
$repository = 'jackzhangqing/cigAMT'
$assetRoot = (Resolve-Path -LiteralPath $AssetDirectory).Path
$command = Get-Command gh -ErrorAction SilentlyContinue
if ($command) { $script:ghExe = $command.Source }
else {
    $script:ghExe = Join-Path $env:ProgramFiles 'GitHub CLI\gh.exe'
    if (-not (Test-Path -LiteralPath $script:ghExe)) { throw 'Install GitHub CLI and run gh auth login first.' }
}

function Invoke-Gh {
    param([string[]]$Arguments, [switch]$AllowFailure)
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $output = @(& $script:ghExe @Arguments 2>&1)
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previousPreference }
    $result = [pscustomobject]@{ Code = $code; Text = (($output | ForEach-Object { [string]$_ }) -join "`n") }
    if ($code -ne 0 -and -not $AllowFailure) { throw $result.Text }
    return $result
}

function Get-WeightRelease {
    for ($attempt = 1; $attempt -le 4; $attempt++) {
        $result = Invoke-Gh -Arguments @('api', "repos/$repository/releases/tags/$Tag") -AllowFailure
        if ($result.Code -eq 0) { return ($result.Text | ConvertFrom-Json) }
        if ($result.Text -match 'HTTP 404') {
            # The authenticated list also finds an interrupted draft upload.
            $list = Invoke-Gh -Arguments @('api', "repos/$repository/releases?per_page=100")
            $found = @(foreach ($item in ($list.Text | ConvertFrom-Json)) {
                if ($item.tag_name -eq $Tag) { $item }
            })
            if ($found.Count -gt 1) { throw 'Multiple releases match the tag.' }
            if ($found.Count -eq 1) { return $found[0] }
            return $null
        }
        if ($attempt -eq 4) { throw $result.Text }
        Start-Sleep -Seconds (2 * $attempt)
    }
}

function Find-RemoteAsset {
    param($Release, $Expected)
    $found = @($Release.assets | Where-Object { $_.name -ceq $Expected.name })
    if ($found.Count -gt 1) { throw "Duplicate remote asset name: $($Expected.name)" }
    if ($found.Count -eq 0) { return $null }
    return $found[0]
}

function Assert-MatchingAsset {
    param($Remote, $Expected)
    if ($Remote.state -ne 'uploaded' -or [int64]$Remote.size -ne [int64]$Expected.bytes) {
        throw "Remote asset incomplete or size differs: $($Expected.name). Existing content was not overwritten."
    }
    if (-not $Remote.digest) { throw "Remote SHA-256 not available yet: $($Expected.name). Retry later; no overwrite was attempted." }
    if ($Remote.digest.ToLowerInvariant() -ne ('sha256:' + $Expected.sha256.ToLowerInvariant())) {
        throw "Remote SHA-256 differs: $($Expected.name). Existing content was not overwritten."
    }
}

$localFiles = @{}
foreach ($asset in $manifest.assets) {
    $path = Join-Path $assetRoot $asset.name
    $file = Get-Item -LiteralPath $path
    if ($file.Length -ne $asset.bytes) { throw "Size mismatch: $($asset.name)" }
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $asset.sha256) {
        throw "Checksum mismatch: $($asset.name)"
    }
    $localFiles[$asset.name] = $path
    Write-Host "Local SHA-256 verified: $($asset.name)"
}
$login = Invoke-Gh -Arguments @('api', 'user', '--jq', '.login')
if ($login.Text.Trim() -cne 'jackzhangqing') { throw 'Authenticate GitHub CLI as jackzhangqing first.' }
$release = Get-WeightRelease
if (-not $release) {
    Write-Host "Creating draft release $Tag"
    $null = Invoke-Gh -Arguments @('release', 'create', $Tag, '--repo', $repository, '--target', 'main', '--draft',
        '--title', 'cigAMT pretrained models', '--notes-file', (Join-Path $repoRoot 'code/WEIGHTS_RELEASE_NOTES.md'))
    $release = Get-WeightRelease
    if (-not $release) { throw 'Could not find the newly created draft. Inspect GitHub before retrying.' }
} else { Write-Host "Continuing existing release $Tag" }

$index = 0
foreach ($asset in $manifest.assets) {
    $index++
    $release = Get-WeightRelease
    $remote = Find-RemoteAsset -Release $release -Expected $asset
    if ($remote -and $remote.state -eq 'uploaded') {
        Assert-MatchingAsset -Remote $remote -Expected $asset
        Write-Host "[$index/$($manifest.assets.Count)] Already verified, skipping: $($asset.name)"
        continue
    }
    if ($remote) {
        # A failed draft upload can leave an empty API placeholder, never a usable asset.
        if ($release.draft -and $remote.state -eq 'starter' -and [int64]$remote.size -eq 0) {
            $null = Invoke-Gh -Arguments @('api', '--method', 'DELETE', "repos/$repository/releases/assets/$($remote.id)")
        } else { throw "Unexpected existing asset: $($asset.name). No content was overwritten." }
    }
    $verified = $false
    for ($attempt = 1; $attempt -le 4; $attempt++) {
        Write-Host "[$index/$($manifest.assets.Count)] Uploading $($asset.name) ($([math]::Round($asset.bytes / 1MB, 1)) MiB), attempt $attempt"
        $upload = Invoke-Gh -Arguments @('release', 'upload', $Tag, $localFiles[$asset.name], '--repo', $repository) -AllowFailure
        $release = Get-WeightRelease
        $remote = Find-RemoteAsset -Release $release -Expected $asset
        if ($remote -and $remote.state -eq 'uploaded') {
            Assert-MatchingAsset -Remote $remote -Expected $asset
            Write-Host "Remote SHA-256 verified: $($asset.name)"
            $verified = $true
            break
        }
        if ($remote -and $release.draft -and $remote.state -eq 'starter' -and [int64]$remote.size -eq 0) {
            $null = Invoke-Gh -Arguments @('api', '--method', 'DELETE', "repos/$repository/releases/assets/$($remote.id)")
        } elseif ($remote) { throw "Incomplete remote asset retained: $($asset.name). Inspect the draft." }
        if ($attempt -eq 4) { throw "Upload incomplete: $($asset.name). Re-run the same command to skip verified packages. $($upload.Text)" }
        Start-Sleep -Seconds (3 * $attempt)
    }
    if (-not $verified) { throw "Verification failed: $($asset.name)" }
}

$release = Get-WeightRelease
foreach ($asset in $manifest.assets) {
    $remote = Find-RemoteAsset -Release $release -Expected $asset
    if (-not $remote) { throw "Missing remote asset: $($asset.name)" }
    Assert-MatchingAsset -Remote $remote -Expected $asset
}
if ($release.draft) {
    Write-Host 'All expected packages verified. Publishing the release.'
    $null = Invoke-Gh -Arguments @('release', 'edit', $Tag, '--repo', $repository, '--draft=false')
}
$release = Get-WeightRelease
if ($release.draft) { throw 'Release is still a draft.' }
Write-Host "SUCCESS: $($manifest.assets.Count) packages remotely verified."
Write-Host $release.html_url

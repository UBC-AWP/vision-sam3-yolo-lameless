param(
  [string[]]$Services = @("admin-frontend", "admin-backend", "video-ingestion", "video-preprocessing", "postgres", "nats", "qdrant")
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Test-GpuAvailable {
  try {
    $output = & nvidia-smi 2>$null
    return $LASTEXITCODE -eq 0
  } catch {
    return $false
  }
}

$useGpu = Test-GpuAvailable
$profile = if ($useGpu) { "gpu" } else { "cpu" }
$composeFile = "docker-compose.yml"
$env:COMPOSE_PROFILES = $profile

$resolvedServices = @()
foreach ($service in $Services) {
  if ($service -eq "video-preprocessing") {
    $resolvedServices += $(if ($useGpu) { "video-preprocessing" } else { "video-preprocessing-cpu" })
  } else {
    $resolvedServices += $service
  }
}

Write-Host ("Using {0} (GPU={1}, profile={2})" -f $composeFile, $useGpu, $profile)
if (-not $useGpu) {
  Write-Host "No NVIDIA GPU detected. Falling back to CPU preprocessing service."
}

if ($Services.Count -eq 0) {
  docker compose -f $composeFile up --build
} else {
  docker compose -f $composeFile up --build @resolvedServices
}

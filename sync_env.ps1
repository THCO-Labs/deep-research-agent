# Sync all .env secrets and environment variables to Azure Container App

$RESOURCE_GROUP = "rg-deepresearch-bench"
$CONTAINER_APP = "app-drbench-api"

if (-not (Test-Path .env)) {
    Write-Error ".env file not found!"
    exit 1
}

$lines = Get-Content .env
$secretArgs = @()
$envArgs = @()

foreach ($line in $lines) {
    $line = $line.Trim()
    if ($line.StartsWith("#") -or [string]::IsNullOrWhiteSpace($line)) { continue }
    $parts = $line.Split("=", 2)
    if ($parts.Count -eq 2) {
        $key = $parts[0].Trim()
        $val = $parts[1].Trim()
        $secretName = ($key.ToLower() -replace "_", "-").Replace("[^a-z0-9-]", "")
        
        $secretArgs += "$secretName=$val"
        $envArgs += "$key=secretref:$secretName"
    }
}

$envArgs += "RUNS_DIR=/app/runs"

Write-Host "Setting $( $secretArgs.Count ) secrets on $CONTAINER_APP..." -ForegroundColor Green
az containerapp secret set --name $CONTAINER_APP --resource-group $RESOURCE_GROUP --secrets $secretArgs

Write-Host "Setting $( $envArgs.Count ) environment variables on $CONTAINER_APP..." -ForegroundColor Green
az containerapp update --name $CONTAINER_APP --resource-group $RESOURCE_GROUP --set-env-vars $envArgs

Write-Host "Sync complete!" -ForegroundColor Green

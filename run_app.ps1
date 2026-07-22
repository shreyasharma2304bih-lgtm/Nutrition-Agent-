$env:WATSONX_API_KEY    = "2WO9GZtrw8Y7cIKfCxkJEUw2EJ_A9xYBGSu8pG_fjXPR"
$env:WATSONX_PROJECT_ID = "63ee0dc8-dfef-470d-82fa-03d0b5551cf7"
$env:WATSONX_URL        = "https://us-south.ml.cloud.ibm.com"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  NutriWise AI - Personalized Nutrition Coach" -ForegroundColor Green
Write-Host "  Powered by IBM watsonx.ai Granite Models" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  WATSONX_API_KEY    : SET" -ForegroundColor Cyan
Write-Host "  WATSONX_PROJECT_ID : $env:WATSONX_PROJECT_ID" -ForegroundColor Cyan
Write-Host "  WATSONX_URL        : $env:WATSONX_URL" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Starting Flask server at http://127.0.0.1:5000" -ForegroundColor Yellow
Write-Host "  Press Ctrl+C to stop." -ForegroundColor Yellow
Write-Host ""

python app.py

param(
  [string]$OutputRoot = (Join-Path (Join-Path $PSScriptRoot "..") "dist"),
  [string]$BundleName = "DicomVisionServer"
)

$ErrorActionPreference = "Stop"

$serverRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$pythonPath = Join-Path $serverRoot ".venv\Scripts\python.exe"
$desktopEntryPath = Join-Path $serverRoot "desktop_entry.py"
$runEntryPath = Join-Path $serverRoot "run.py"
$outputRootPath = [System.IO.Path]::GetFullPath($OutputRoot)
$workRoot = Join-Path $serverRoot "build\pyinstaller"
$bundleRoot = Join-Path $outputRootPath $BundleName
$bundleExecutable = Join-Path $bundleRoot "$BundleName.exe"

if (!(Test-Path $pythonPath)) {
  throw "Python executable not found: $pythonPath"
}

$entryPath =
  if (Test-Path $desktopEntryPath) {
    $desktopEntryPath
  }
  elseif (Test-Path $runEntryPath) {
    $runEntryPath
  }
  else {
    throw "Server entry not found. Checked: $desktopEntryPath, $runEntryPath"
  }

New-Item -ItemType Directory -Force -Path $outputRootPath | Out-Null
New-Item -ItemType Directory -Force -Path $workRoot | Out-Null

if (Test-Path $bundleRoot) {
  Remove-Item -LiteralPath $bundleRoot -Recurse -Force
}

Push-Location $serverRoot
try {
  & $pythonPath -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --name $BundleName `
    --distpath $outputRootPath `
    --workpath $workRoot `
    --specpath $workRoot `
    --hidden-import scipy `
    --hidden-import scipy.ndimage `
    --hidden-import scipy.ndimage._nd_image `
    --exclude-module matplotlib `
    --exclude-module pytest `
    --exclude-module scipy.conftest `
    --exclude-module scipy.tests `
    --hidden-import vtkmodules.util.numpy_support `
    --hidden-import vtkmodules.util.vtkConstants `
    --hidden-import vtkmodules.vtkCommonCore `
    --hidden-import vtkmodules.vtkCommonDataModel `
    --hidden-import vtkmodules.vtkRenderingCore `
    --hidden-import vtkmodules.vtkRenderingOpenGL2 `
    --hidden-import vtkmodules.vtkRenderingVolumeOpenGL2 `
    --collect-binaries vtk `
    --collect-data vtk `
    $entryPath

  if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
  }

  if (!(Test-Path $bundleExecutable)) {
    throw "Bundle build completed without expected executable: $bundleExecutable"
  }

  Write-Output "Desktop bundle built at: $bundleRoot"
}
finally {
  Pop-Location
}

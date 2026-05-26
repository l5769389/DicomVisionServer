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
$pyInstallerRequirement = "pyinstaller>=6.11.0,<7.0.0"

if (!(Test-Path $pythonPath)) {
  throw "Python executable not found: $pythonPath"
}

$hasPyInstaller = $false
& $pythonPath -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('PyInstaller') else 1)"
if ($LASTEXITCODE -eq 0) {
  $hasPyInstaller = $true
}
else {
  $uvCommand = Get-Command uv -ErrorAction SilentlyContinue
  if ($null -eq $uvCommand) {
    throw "PyInstaller is not installed in $pythonPath and uv was not found. Install it with: uv pip install '$pyInstallerRequirement'"
  }
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
  $pyInstallerArgs = @(
    "--noconfirm",
    "--clean",
    "--onedir",
    "--name",
    $BundleName,
    "--distpath",
    $outputRootPath,
    "--workpath",
    $workRoot,
    "--specpath",
    $workRoot,
    "--hidden-import",
    "scipy",
    "--hidden-import",
    "scipy.ndimage",
    "--hidden-import",
    "scipy.ndimage._nd_image",
    "--exclude-module",
    "matplotlib",
    "--exclude-module",
    "pytest",
    "--exclude-module",
    "scipy.conftest",
    "--exclude-module",
    "scipy.tests",
    "--hidden-import",
    "vtkmodules.util.numpy_support",
    "--hidden-import",
    "vtkmodules.util.vtkConstants",
    "--hidden-import",
    "vtkmodules.vtkCommonCore",
    "--hidden-import",
    "vtkmodules.vtkCommonDataModel",
    "--hidden-import",
    "vtkmodules.vtkRenderingCore",
    "--hidden-import",
    "vtkmodules.vtkRenderingOpenGL2",
    "--hidden-import",
    "vtkmodules.vtkRenderingVolumeOpenGL2",
    "--collect-binaries",
    "vtk",
    "--collect-data",
    "vtk",
    $entryPath
  )

  if ($hasPyInstaller) {
    & $pythonPath -m PyInstaller @pyInstallerArgs
  }
  else {
    & uv run --with $pyInstallerRequirement python -m PyInstaller @pyInstallerArgs
  }

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

# This script compiles mumax3 for Windows 10 against a specified CUDA version.

param(
    [String[]]$CUDA_VERSIONS = ("12.6"), # Only versions you have installed
    [Int[]]$CUDA_CC,
    [String[]]$CUDA_KERNELS
)

foreach ($CUDA_VERSION_STR in $CUDA_VERSIONS) {

    # Reset CUDA_CC each iteration to avoid carrying over previous loop values
    $CUDA_CC = $null

    # Final build directory
    $builddir = "build/mumax3_windows_cuda$CUDA_VERSION_STR"

    # Map CUDA version to environment variable
    switch ($CUDA_VERSION_STR) {
        "9.2"  { $CUDA_HOME = $env:CUDA_PATH_V9_2  }
        "10.0" { $CUDA_HOME = $env:CUDA_PATH_V10_0 }
        "10.1" { $CUDA_HOME = $env:CUDA_PATH_V10_1 }
        "10.2" { $CUDA_HOME = $env:CUDA_PATH_V10_2 }
        "12.6" { $CUDA_HOME = $env:CUDA_PATH_V12_6 }
        default { Write-Output "Unsupported CUDA version $CUDA_VERSION_STR"; exit }
    }

    if (-not $CUDA_HOME -or (-not (Test-Path $CUDA_HOME))) {
        Write-Output "CUDA version $CUDA_VERSION_STR does not seem to be installed"
        exit
    }

    # Map CUDA version to compute capabilities (match working setup)
    if (-not $CUDA_CC) {
        switch ($CUDA_VERSION_STR) {
            "9.2"  { $CUDA_CC = 30,32,35,37,50,52,53,60,61,62,70,72 }
            "10.0" { $CUDA_CC = 30,32,35,37,50,52,53,60,61,62,70,72,75 }
            "10.1" { $CUDA_CC = 30,32,35,37,50,52,53,60,61,62,70,72,75 }
            "10.2" { $CUDA_CC = 30,32,35,37,50,52,53,60,61,62,70,72,75 }
            "12.6" { $CUDA_CC = 61 }  # Only compile for this CC to match working setup
            default { Write-Output "No compute capability defined for $CUDA_VERSION_STR"; exit }
        }
    }

    # Visual Studio paths (must match your installed version)
    $VS2022 = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\bin\Hostx86\x64"
    $VS2017 = "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64"

    # Pick compiler based on CUDA version
    $CUDA_VERSION = [Version]::Parse($CUDA_VERSION_STR)
    if ($CUDA_VERSION -lt [Version]::new(11.6)) {
        $CCBIN = $VS2017
    } else {
        $CCBIN = if (Test-Path $VS2022) { $VS2022 } else { $VS2017 }
    }

    if (-not (Test-Path $CCBIN)) {
        Write-Output "CCBIN for nvcc not found at $CCBIN"
        exit
    }

    # NVIDIA compiler
    $NVCC = "${CUDA_HOME}/bin/nvcc.exe"

    # CGO flags
    $env:CGO_LDFLAGS = "-lcufft -lcurand -lcuda -L${CUDA_HOME}/lib/x64"
    $env:CGO_CFLAGS = "-I${CUDA_HOME}/include -w"

    # Compile CUDA kernels
    Set-Location ..\cuda
        Remove-Item *.ptx -ErrorAction Ignore
        Remove-Item *_wrapper.go -ErrorAction Ignore

        go build -v .\cuda2go.go

        if ($CUDA_KERNELS.Length -eq 0) {
            $cudafiles = Get-ChildItem -Filter "*.cu"
        } else {
            $cudafiles = Get-ChildItem -Filter "*.cu" | Where-Object { $CUDA_KERNELS -contains $_.BaseName }
        }

        foreach ($cudafile in $cudafiles) {
            $kernel = $cudafile.BaseName
            Remove-Item "${kernel}_*.ptx" -ErrorAction Ignore
            Remove-Item "${kernel}_*wrapper.go" -ErrorAction Ignore
            foreach ($cc in $CUDA_CC) {
                & $NVCC -ccbin "${CCBIN}" -Xptxas -O3 -ptx `
                    -gencode="arch=compute_${cc},code=sm_${cc}" `
                    "${cudafile}" -o "${kernel}_${cc}.ptx"
            }
            & .\cuda2go $cudafile
        }

    Set-Location ..

    # Compile mumax3 executables
    $COMMIT_HASH = git rev-parse --short HEAD 2>$null
    if (-not $COMMIT_HASH) { $COMMIT_HASH = "unknown" }

    go install -ldflags "-X main.commitHash=$COMMIT_HASH" -v "github.com/mumax/3/..."

    # Copy executables and CUDA DLLs to build dir
    Remove-Item -Recurse -Force $builddir -ErrorAction Ignore
    New-Item -ItemType Directory -Force $builddir

    foreach ($app in "mumax3.exe","mumax3-convert.exe","mumax3-server.exe") {
        Copy-Item "${env:GOPATH}/bin/${app}" -Destination $builddir
    }
    Copy-Item "${CUDA_HOME}/bin/cufft64*.dll" -Destination $builddir
    Copy-Item "${CUDA_HOME}/bin/curand64*.dll" -Destination $builddir

    # Optional: compress into zip
    Remove-Item -Force "${builddir}.zip" -ErrorAction Ignore
    Compress-Archive -Path "${builddir}/*" -DestinationPath "${builddir}.zip"

    Write-Output "Build for CUDA $CUDA_VERSION_STR complete at $builddir"
}
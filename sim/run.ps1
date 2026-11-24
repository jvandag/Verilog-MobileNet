param([string]$VivadoBin=$env:VIVADO_BIN, [int]$Count=0, [switch]$PrimitiveInference)
$ErrorActionPreference='Stop'
if (!$VivadoBin) { throw 'Set VIVADO_BIN to the Vivado bin directory' }
$root=Split-Path $PSScriptRoot
$build=Join-Path $root 'build'
if(!$Count) {$Count=[int](Get-Content "$build/fixtures/count.txt")}
New-Item -ItemType Directory -Force $build | Out-Null
Push-Location $build
try {
    $romSnapshot="rom_sim_$PID"
    $cnnSnapshot="cnn_sim_$PID"
    Copy-Item "$root/rtl/generated/*.mem" . -Force
    & "$VivadoBin/xvlog.bat" -sv -i "$root/rtl/generated" "$root/rtl/cnn_weights.sv" "$root/sim/tb_cnn_rom.sv" "$VivadoBin/../data/verilog/src/glbl.v"
    if($LASTEXITCODE) {throw 'ROM compilation failed'}
    & "$VivadoBin/xelab.bat" tb_cnn_rom glbl -s $romSnapshot -L unisims_ver -mt off
    if($LASTEXITCODE) {throw 'ROM elaboration failed'}
    & "$VivadoBin/xsim.bat" $romSnapshot -runall -log rom_test.log
    if($LASTEXITCODE -or !(Select-String -Path rom_test.log -Pattern 'ROM TEST PASSED' -Quiet)) {throw 'physical ROM test failed'}
    $defines=@()
    if(!$PrimitiveInference) {$defines=@('-d','CNN_FAST_SIM')}
    & "$VivadoBin/xvlog.bat" -sv @defines -i "$root/rtl/generated" "$root/rtl/cnn_ram.sv" "$root/rtl/cnn_weights.sv" "$root/rtl/cnn_core.sv" "$root/rtl/wb_cnn.sv" "$root/sim/tb_wb_cnn.sv"
    if($LASTEXITCODE) {throw 'xvlog failed'}
    & "$VivadoBin/xvlog.bat" "$VivadoBin/../data/verilog/src/glbl.v"
    if($LASTEXITCODE) {throw 'glbl compilation failed'}
    Copy-Item "$root/rtl/generated/*.mem" . -Force
    & "$VivadoBin/xelab.bat" tb_wb_cnn glbl -s $cnnSnapshot -debug typical -L unisims_ver
    if($LASTEXITCODE) {throw 'xelab failed'}
    & "$VivadoBin/xsim.bat" $cnnSnapshot -runall -testplusarg "`"COUNT=$Count`""
    if($LASTEXITCODE -or !(Select-String -Path xsim.log -Pattern 'TEST PASSED' -Quiet)) {throw 'xsim failed'}
} finally {Pop-Location}

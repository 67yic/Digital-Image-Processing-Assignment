$p = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*age_estimation_pro.py*' } | Select-Object -First 1
if (-not $p) { Write-Output 'NO_PROCESS'; exit 0 }
try {
	$proc = Get-Process -Id $p.ProcessId -ErrorAction Stop
	$start = $proc.StartTime
	$elapsed = (Get-Date) - $start
	Write-Output ("PID:$($p.ProcessId)")
	Write-Output ("START:" + $start.ToString("s"))
	Write-Output ("ELAPSED_SECONDS:" + [int]$elapsed.TotalSeconds)
} catch {
	Write-Output "PID:$($p.ProcessId)"
	Write-Output 'START:UNKNOWN'
	Write-Output 'ELAPSED_SECONDS:0'
}

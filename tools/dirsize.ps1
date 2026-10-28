param([string]$Root, [switch]$Children, [double]$TopN = 0)

$ErrorActionPreference = 'SilentlyContinue'

function Get-DirStats([string]$Path) {
    $total = [long]0
    $fileCount = [long]0
    $stack = [System.Collections.Generic.Stack[string]]::new()
    $stack.Push($Path)
    while ($stack.Count -gt 0) {
        $cur = $stack.Pop()
        try {
            $di = [System.IO.DirectoryInfo]::new($cur)
            foreach ($e in $di.EnumerateFileSystemInfos()) {
                $attr = $e.Attributes
                if ($attr -band [System.IO.FileAttributes]::ReparsePoint) { continue }
                if ($attr -band [System.IO.FileAttributes]::Directory) {
                    $stack.Push($e.FullName)
                } else {
                    $total += $e.Length
                    $fileCount++
                }
            }
        } catch { }
    }
    return @{ Bytes = $total; Files = $fileCount }
}

if ($Children) {
    $rows = foreach ($d in (Get-ChildItem -LiteralPath $Root -Force -Directory)) {
        $s = Get-DirStats $d.FullName
        [pscustomobject]@{
            GB   = [math]::Round($s.Bytes / 1GB, 3)
            Files = $s.Files
            Path = $d.FullName
        }
    }
    $sorted = $rows | Sort-Object GB -Descending
    if ($TopN -gt 0) { $sorted | Select-Object -First $TopN } else { $sorted }
} else {
    $s = Get-DirStats $Root
    [pscustomobject]@{ GB = [math]::Round($s.Bytes / 1GB, 3); Files = $s.Files; Path = $Root }
}

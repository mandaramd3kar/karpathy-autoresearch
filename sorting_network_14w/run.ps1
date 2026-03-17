param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $RemainingArgs
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
python (Join-Path $scriptDir 'experiment.py') @RemainingArgs

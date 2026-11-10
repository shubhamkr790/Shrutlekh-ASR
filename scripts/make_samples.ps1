# Generates the evaluation clips in samples/ using the speech voices built into Windows.
# Each clip has a known reference text (samples/references.json), so word error rate
# can be measured exactly. Run from the project root:
#   powershell -ExecutionPolicy Bypass -File scripts/make_samples.ps1

Add-Type -AssemblyName System.Speech
$root = Split-Path -Parent $PSScriptRoot
$out = Join-Path $root "samples"
New-Item -ItemType Directory -Force -Path $out | Out-Null

$refs = Get-Content (Join-Path $out "references.json") -Raw | ConvertFrom-Json
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)

foreach ($clip in $refs.clips) {
    if ($clip.generated) { continue }  # built by make_noisy_sample.py instead
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $synth.SelectVoice($clip.voice)
    $synth.Rate = $clip.rate
    $path = Join-Path $out $clip.file
    $synth.SetOutputToWaveFile($path, $format)
    $synth.Speak($clip.text)
    $synth.Dispose()
    Write-Host "wrote $($clip.file)"
}

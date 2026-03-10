param(
    [string]$InputPath = "outputs/formal_gsm8k_qwen3_8b_1/records.jsonl",
    [string]$OutDir = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $parent = Split-Path -Parent $InputPath
    $OutDir = Join-Path $parent "cleaned_analysis"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$numberPattern = '-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?%?'
$referencePattern = '####\s*(.+)$'
$finalPattern = '(?is)Final\s*Answer\s*:\s*(.+)$'
$stepPattern = '(?im)^\s*Step\s*\d+\s*:'
$selfRevisionPattern = '(?i)\bwait\b|let me check|let me think|hold on'
$corruptionPattern = '极|朋友|许多|æ|ï¼|Ã'

function Get-NormalizedAnswer {
    param([AllowNull()][string]$Text)

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return ""
    }

    $clean = $Text.Trim()
    $clean = $clean.Replace("\boxed{", "").Replace("}", "")
    $clean = $clean.Replace(",", "").Replace("$", "")

    $matches = [regex]::Matches($clean, $numberPattern)
    if ($matches.Count -gt 0) {
        $value = $matches[$matches.Count - 1].Value
        if (($value -notmatch '/') -and ($value -notmatch '%')) {
            try {
                $num = [double]$value
                if ([math]::Abs($num - [math]::Round($num)) -lt 1e-9) {
                    return ([int64][math]::Round($num)).ToString()
                }
                return $num.ToString("G17")
            } catch {
                return $value
            }
        }
        return $value
    }

    return ([regex]::Replace($clean.ToLower(), '\s+', ' ')).Trim()
}

function Get-ReferenceAnswer {
    param([string]$ReferenceSolution)

    $m = [regex]::Match($ReferenceSolution, $referencePattern, [System.Text.RegularExpressions.RegexOptions]::Singleline)
    if ($m.Success) {
        return Get-NormalizedAnswer $m.Groups[1].Value
    }
    return Get-NormalizedAnswer $ReferenceSolution
}

function Get-StudentFinalAnswer {
    param([string]$StudentSolution)

    $m = [regex]::Match($StudentSolution, $finalPattern)
    if ($m.Success) {
        return Get-NormalizedAnswer $m.Groups[1].Value
    }
    return ""
}

function Get-StudentTailAnswer {
    param([string]$StudentSolution)

    $lines = @($StudentSolution -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($lines.Count -eq 0) {
        return ""
    }
    return Get-NormalizedAnswer $lines[$lines.Count - 1]
}

function Has-CorrectAnswerAnywhere {
    param(
        [string]$StudentSolution,
        [string]$ReferenceAnswer
    )

    if ([string]::IsNullOrWhiteSpace($ReferenceAnswer)) {
        return $false
    }

    $escaped = [regex]::Escape($ReferenceAnswer)
    return [regex]::IsMatch($StudentSolution.Replace(",", ""), "(?<!\d)$escaped(?!\d)")
}

$records = @()
Get-Content $InputPath | ForEach-Object {
    if (-not [string]::IsNullOrWhiteSpace($_)) {
        $records += ($_ | ConvertFrom-Json)
    }
}

$cleaned = New-Object System.Collections.Generic.List[object]
$categoryCounts = [ordered]@{
    correct_original = 0
    recovered_correct = 0
    wrong_analyzable = 0
    wrong_unanalyzable = 0
    wrong_format_noise = 0
}
$categoryExamples = @{
    correct_original = New-Object System.Collections.Generic.List[string]
    recovered_correct = New-Object System.Collections.Generic.List[string]
    wrong_analyzable = New-Object System.Collections.Generic.List[string]
    wrong_unanalyzable = New-Object System.Collections.Generic.List[string]
    wrong_format_noise = New-Object System.Collections.Generic.List[string]
}

foreach ($record in $records) {
    $studentSolution = [string]$record.student_solution
    $referenceAnswer = Get-ReferenceAnswer ([string]$record.reference_solution)
    $studentFinal = Get-StudentFinalAnswer $studentSolution
    $studentTail = Get-StudentTailAnswer $studentSolution
    $stepCount = @($record.step_kls).Count

    $hasFinal = [regex]::IsMatch($studentSolution, $finalPattern)
    $singleStep = $stepCount -le 1
    $selfRevisionLoop = [regex]::IsMatch($studentSolution, $selfRevisionPattern)
    $encodingNoise = [regex]::IsMatch($studentSolution, $corruptionPattern)
    $containsReferenceAnswer = Has-CorrectAnswerAnywhere $studentSolution $referenceAnswer
    $finalAnswerMatches = (-not [string]::IsNullOrWhiteSpace($studentFinal)) -and ($studentFinal -eq $referenceAnswer)
    $tailAnswerMatches = (-not $hasFinal) -and (-not [string]::IsNullOrWhiteSpace($studentTail)) -and ($studentTail -eq $referenceAnswer)

    $recoveredCorrect = $false
    $recoveryReason = $null

    if (-not [bool]$record.is_correct) {
        if ($finalAnswerMatches) {
            $recoveredCorrect = $true
            $recoveryReason = "final_answer_matches_reference"
        } elseif ($tailAnswerMatches -and $stepCount -ge 2 -and -not $selfRevisionLoop) {
            $recoveredCorrect = $true
            $recoveryReason = "tail_answer_matches_reference"
        }
    }

    $cleanedIsCorrect = [bool]$record.is_correct -or $recoveredCorrect
    $wrongNoise = (-not $cleanedIsCorrect) -and (
        (-not $hasFinal) -or
        $singleStep -or
        $selfRevisionLoop -or
        $encodingNoise
    )
    $analyzableWrong = (-not $cleanedIsCorrect) -and (-not $wrongNoise)

    if ([bool]$record.is_correct) {
        $category = "correct_original"
    } elseif ($recoveredCorrect) {
        $category = "recovered_correct"
    } elseif ($analyzableWrong) {
        $category = "wrong_analyzable"
    } elseif ((-not $hasFinal) -or $encodingNoise) {
        $category = "wrong_format_noise"
    } else {
        $category = "wrong_unanalyzable"
    }

    $categoryCounts[$category]++
    if ($categoryExamples[$category].Count -lt 10) {
        $categoryExamples[$category].Add([string]$record.id)
    }

    $flags = [ordered]@{
        missing_final = -not $hasFinal
        single_step = $singleStep
        self_revision_loop = $selfRevisionLoop
        encoding_noise = $encodingNoise
        contains_reference_answer_anywhere = $containsReferenceAnswer
        final_answer_matches_reference = $finalAnswerMatches
        tail_answer_matches_reference = $tailAnswerMatches
    }

    $cleanedRecord = [ordered]@{}
    foreach ($prop in $record.PSObject.Properties) {
        $cleanedRecord[$prop.Name] = $prop.Value
    }
    $cleanedRecord["clean_reference_answer"] = $referenceAnswer
    $cleanedRecord["clean_student_final_answer"] = $studentFinal
    $cleanedRecord["clean_student_tail_answer"] = $studentTail
    $cleanedRecord["cleaned_is_correct"] = $cleanedIsCorrect
    $cleanedRecord["cleaned_category"] = $category
    $cleanedRecord["cleaning_flags"] = $flags
    $cleanedRecord["recovery_reason"] = $recoveryReason
    $cleanedRecord["is_analyzable_wrong"] = $analyzableWrong

    $cleaned.Add([pscustomobject]$cleanedRecord)
}

$cleanedPath = Join-Path $OutDir "records.cleaned.jsonl"
if (Test-Path $cleanedPath) {
    Remove-Item $cleanedPath
}
foreach ($row in $cleaned) {
    Add-Content -Path $cleanedPath -Value ($row | ConvertTo-Json -Depth 8 -Compress)
}

$analyzableWrongRows = @($cleaned | Where-Object { $_.is_analyzable_wrong })
$analyzablePath = Join-Path $OutDir "wrong_for_jump_analysis.jsonl"
if (Test-Path $analyzablePath) {
    Remove-Item $analyzablePath
}
foreach ($row in $analyzableWrongRows) {
    Add-Content -Path $analyzablePath -Value ($row | ConvertTo-Json -Depth 8 -Compress)
}

$recoveredRows = @($cleaned | Where-Object { $_.cleaned_category -eq "recovered_correct" })
$recoveredPath = Join-Path $OutDir "recovered_correct.jsonl"
if (Test-Path $recoveredPath) {
    Remove-Item $recoveredPath
}
foreach ($row in $recoveredRows) {
    Add-Content -Path $recoveredPath -Value ($row | ConvertTo-Json -Depth 8 -Compress)
}

$summary = [ordered]@{
    input_path = $InputPath
    n_total = $cleaned.Count
    n_original_correct = @($cleaned | Where-Object { $_.is_correct }).Count
    n_original_wrong = @($cleaned | Where-Object { -not $_.is_correct }).Count
    n_cleaned_correct = @($cleaned | Where-Object { $_.cleaned_is_correct }).Count
    n_cleaned_wrong = @($cleaned | Where-Object { -not $_.cleaned_is_correct }).Count
    n_recovered_correct = $recoveredRows.Count
    n_analyzable_wrong = $analyzableWrongRows.Count
    n_wrong_format_noise = @($cleaned | Where-Object { $_.cleaned_category -eq "wrong_format_noise" }).Count
    n_wrong_unanalyzable = @($cleaned | Where-Object { $_.cleaned_category -eq "wrong_unanalyzable" }).Count
    category_counts = $categoryCounts
    category_examples = @{
        correct_original = @($categoryExamples["correct_original"])
        recovered_correct = @($categoryExamples["recovered_correct"])
        wrong_analyzable = @($categoryExamples["wrong_analyzable"])
        wrong_unanalyzable = @($categoryExamples["wrong_unanalyzable"])
        wrong_format_noise = @($categoryExamples["wrong_format_noise"])
    }
    output_files = @{
        cleaned_records = $cleanedPath
        recovered_correct = $recoveredPath
        wrong_for_jump_analysis = $analyzablePath
    }
}

$summaryPath = Join-Path $OutDir "cleaned_summary.json"
$summary | ConvertTo-Json -Depth 8 | Set-Content -Path $summaryPath -Encoding UTF8

Write-Output ($summary | ConvertTo-Json -Depth 8)

# RoboGen Changes

## 2026-01-02 - Validation and Retry Logic for LLM Parsing

### Summary
Added comprehensive validation and automatic retry logic to all LLM response parsing functions to handle cases where the model doesn't return responses in the expected format.

### Problem
Previously, when LLM responses didn't match the expected format, the code would:
- Silently create empty files (e.g., empty `substeps.txt`)
- Skip generating required `.py` files
- Continue execution without warnings or errors
- Make debugging difficult since failures were invisible

### Solution
Implemented a robust retry mechanism with validation checks:

1. **Automatic Retries**: Up to 5 attempts for each LLM query if parsing fails
2. **Validation Checks**: Explicit validation that required fields are present and non-empty
3. **Clear Error Messages**: Informative error messages showing what failed and response previews
4. **Progress Logging**: Visual feedback (✓/❌) showing parsing success or failure

### Changes Made

#### 1. Substep Decomposition (`gpt_4/prompts/prompt_manipulation_reward_primitive.py`)
- **Function**: `decompose_and_generate_reward_or_primitive()`
- **Validation Added**:
  - Checks that substeps list is not empty
  - Validates that substeps count matches substep_types count
  - Validates that substeps count matches reward_or_primitives count
- **Retry Logic**: Up to 5 attempts with detailed error logging
- **Error Feedback**: Shows validation errors and response preview on failure
- **New Parameter**: `max_retries=5` (configurable)

#### 2. YAML Config Generation (`gpt_4/prompts/utils.py`)
- **Function**: `parse_response_to_get_yaml()`
- **Validation Added**:
  - Raises `ValueError` if no YAML block (` ```yaml`) found in response
  - Raises `ValueError` if YAML block is empty
  - Shows response preview in error message
- **Retry Logic**: Added in `build_task_given_text()` caller (up to 5 attempts)
- **Error Handling**: Try-catch wrapper with informative error messages

#### 3. Task Expansion (`gpt_4/prompts/prompt_from_description.py`)
- **Function**: `parse_response()`
- **Validation Added**:
  - Raises `ValueError` if "Description:" field not found
  - Raises `ValueError` if "Additional Objects:" field not found
  - Raises `ValueError` if "Links:" section not found
  - Raises `ValueError` if "Joints:" section not found
  - Shows response preview in error messages
- **Retry Logic**: Added in `expand_task_name()` caller (up to 5 attempts)
- **Error Handling**: Try-catch wrapper with attempt tracking

### Usage

No changes required to existing code! The retry logic is automatic and transparent.

**Optional**: You can customize the number of retries:
```python
# For substep decomposition only (other functions use hardcoded max_retries=5)
decompose_and_generate_reward_or_primitive(
    ...,
    max_retries=10  # Default is 5
)
```

### Example Output

**Success Case:**
```
[OK] Successfully parsed 3 substeps
[OK] Successfully generated and parsed YAML config
[OK] Successfully parsed task expansion
```

**Retry Case:**
```
[FAIL] Parsing validation failed (attempt 1/5):
   - No substeps found in response (expected lines starting with 'Substep:')

Model response preview (first 500 chars):
Here are the steps for the task...

[WARNING] Retry attempt 2/5 for substep decomposition...
[OK] Successfully parsed 3 substeps
```

**Failure After Max Retries:**
```
[FAIL] Parsing validation failed (attempt 5/5):
   - No substeps found in response (expected lines starting with 'Substep:')

ValueError: Failed to parse substeps after 5 attempts. Validation errors:
  - No substeps found in response (expected lines starting with 'Substep:')

Model response:
[full model response shown for debugging]
```

### Benefits

1. **Robustness**: System automatically recovers from transient model output formatting issues
2. **Visibility**: Clear logging shows when retries happen and why
3. **Debuggability**: Detailed error messages help diagnose persistent formatting issues
4. **No Silent Failures**: All parsing failures now raise errors instead of creating empty files
5. **User Experience**: Reduces manual intervention needed when model outputs vary

### Technical Details

- **Import Added**: `warnings` module added to all affected files
- **Exception Type**: Uses `ValueError` for all validation failures
- **Retry Strategy**: Simple retry without backoff (assumes formatting issues, not rate limits)
- **Error Messages**: Include response previews (first 500 chars or 10 lines) for debugging

### Backwards Compatibility

[OK] Fully backwards compatible - no breaking changes to function signatures or return values.

The only behavioral difference is that functions now raise `ValueError` instead of silently returning `None` or empty data when parsing fails after all retries.

### Files Modified

1. `gpt_4/prompts/prompt_manipulation_reward_primitive.py`
   - Added `warnings` import
   - Modified `decompose_and_generate_reward_or_primitive()` with retry loop and validation

2. `gpt_4/prompts/utils.py`
   - Added `warnings` import
   - Modified `parse_response_to_get_yaml()` with validation
   - Modified `build_task_given_text()` with retry loop for YAML generation

3. `gpt_4/prompts/prompt_from_description.py`
   - Added `warnings` import
   - Modified `parse_response()` with validation
   - Modified `expand_task_name()` with retry loop

### Future Improvements

Potential enhancements to consider:
- Add exponential backoff between retries
- Make retry count configurable via environment variable
- Add metrics/logging to track retry rates
- Implement prompt refinement on retry (e.g., add "Please format your response as..." reminder)

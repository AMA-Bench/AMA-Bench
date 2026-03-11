def extract_final_answer(response: str) -> str:
    """Extract the final answer from model response.

    Args:
        response: Raw model response

    Returns:
        Extracted answer string
    """
    # Try to extract answer after "##Answer:" marker
    if "##Answer:" in response:
        parts = response.split("##Answer:")
        if len(parts) > 1:
            # Get everything after ##Answer: and strip whitespace
            answer = parts[1].strip()
            # Remove any trailing markers or extra text
            # Take first line if multi-line
            answer = answer.split('\n')[0].strip()
            return answer

    # If no marker found, return the original response
    return response.strip()

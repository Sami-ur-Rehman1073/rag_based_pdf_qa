# -----------------------------------------------
# split_text_into_chunks()
#
# Arguments:
#   text       : the full extracted PDF text
#   chunk_size : how many characters per chunk
#   overlap    : how many characters to repeat
#                between consecutive chunks
#
# Returns:
#   A list of text chunk strings
#
# Example:
#   chunk_size = 500, overlap = 100
#   Chunk 1 → characters 0   to 500
#   Chunk 2 → characters 400 to 900  (100 overlap)
#   Chunk 3 → characters 800 to 1300 (100 overlap)
# -----------------------------------------------
def split_text_into_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> list[str]:

    # Basic validation
    if not text or not text.strip():
        raise ValueError("Cannot chunk empty text.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks = []

    # -----------------------------------------------
    # We use a sliding window approach:
    # Start at position 0
    # Each iteration moves forward by (chunk_size - overlap)
    # So chunks share `overlap` characters with the next
    # -----------------------------------------------
    start = 0
    total_length = len(text)

    while start < total_length:

        # Calculate end position for this chunk
        end = start + chunk_size

        # Slice the text from start to end
        chunk = text[start:end]

        # Clean up extra whitespace but keep content intact
        chunk = chunk.strip()

        # Only add non-empty chunks
        if chunk:
            chunks.append(chunk)

        # Move the window forward
        # We subtract overlap so next chunk shares
        # some text with the current one
        start += chunk_size - overlap

    return chunks
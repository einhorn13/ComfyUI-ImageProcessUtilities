from itertools import zip_longest

def _parse_ints_from_string(s: str):
    """Parses a comma-separated string into a list of integers, returning None on failure."""
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        return [int(c.strip()) for c in s.split(',')]
    except (ValueError, TypeError):
        print(f"[ERROR] Could not parse coordinates string: '{s}'. Please use comma-separated integers.")
        return None

class StringToIntegers:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"multiline": False, "default": ""})}}
    
    RETURN_TYPES = ("INTS",)
    FUNCTION = "convert"
    CATEGORY = "image/coordinates"

    def convert(self, text: str):
        ints = _parse_ints_from_string(text)
        # Provide a safe default for invalid or empty input to avoid breaking workflows.
        return (ints if ints is not None else [0],)

class CombineCoords:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x_coords": ("INTS", {"default": [0]}),
                "y_coords": ("INTS", {"default": [0]}),
            }
        }
    
    RETURN_TYPES = ("COORDS",)
    FUNCTION = "combine"
    CATEGORY = "image/coordinates"

    def combine(self, x_coords, y_coords):
        # If one list is shorter, fill missing values with the last value of the respective list.
        fill_x = x_coords[-1] if x_coords else 0
        fill_y = y_coords[-1] if y_coords else 0
        
        combined = list(zip_longest(x_coords, y_coords, fillvalue=None))
        result = [[(x if x is not None else fill_x), (y if y is not None else fill_y)] for x, y in combined]
        return (result,)

class SplitCoords:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"coords": ("COORDS",)}}
    
    RETURN_TYPES = ("INTS", "INTS")
    RETURN_NAMES = ("x_coords", "y_coords")
    FUNCTION = "split"
    CATEGORY = "image/coordinates"

    def split(self, coords):
        if not coords or not isinstance(coords, list):
            return ([0], [0])
        
        x_coords = [coord[0] for coord in coords]
        y_coords = [coord[1] for coord in coords]
        return (x_coords, y_coords)
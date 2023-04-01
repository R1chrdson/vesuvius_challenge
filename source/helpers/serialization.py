from json import JSONEncoder
from pathlib import Path


class ConfigEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, Path):
            return str(o)
        if hasattr(o, "__dict__"):
            return o.__dict__
        return JSONEncoder.default(self, o)

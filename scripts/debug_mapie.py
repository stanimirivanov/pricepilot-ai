import sys

sys.path.insert(0, "src")

print("Checking MAPIE installation...")

# 1. Basic import
try:
    import mapie

    print(f"MAPIE version: {mapie.__version__}")
except ImportError as e:
    print(f"Failed to import mapie: {e}")
    sys.exit(1)

# 2. Check mapie module contents
print("\nContents of mapie module:")
print([x for x in dir(mapie) if not x.startswith("_")])

# 3. Try different import paths
import_paths = [
    "from mapie.regression import MapieRegressor",
    "from mapie import MapieRegressor",
    "from mapie.regression.regression import MapieRegressor",
]

print("\nTrying different import paths:")
for path in import_paths:
    try:
        exec(path)
        print(f"✓ {path}")
    except ImportError as e:
        print(f"✗ {path}: {e}")

# 4. Check if mapie.regression exists
try:
    import mapie.regression

    print("\nContents of mapie.regression:")
    print([x for x in dir(mapie.regression) if "Mapie" in x])
except ImportError as e:
    print(f"\nFailed to import mapie.regression: {e}")

# 5. Try mapie.regression.regression
try:
    import mapie.regression.regression

    print("\nContents of mapie.regression.regression:")
    print([x for x in dir(mapie.regression.regression) if "Mapie" in x])
except ImportError as e:
    print(f"\nFailed to import mapie.regression.regression: {e}")

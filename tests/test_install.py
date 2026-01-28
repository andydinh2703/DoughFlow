try: 
    import doughflow
    print("doughflow package is install successfully")
    print(f"  Location: {doughflow.__file__}")
except ImportError as e:
    print(f"Import failed: {e}")
import os
import shutil

if os.path.exists("static"):
    shutil.rmtree("static")
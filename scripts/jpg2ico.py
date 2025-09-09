from PIL import Image
import os

# Ruta de tu JPG dentro de assets
jpg_path = os.path.join("assets", "erreka_logo.jpg")
ico_path = os.path.join("assets", "erreka_logo.ico")

img = Image.open(jpg_path)
# Guarda varias resoluciones para mejor compatibilidad
img.save(ico_path, sizes=[(256,256), (128,128), (64,64), (32,32), (16,16)])
print("Icono generado en", ico_path)

import FreeCAD
import Mesh

doc = FreeCAD.open(r"C:\Users\Yan\Documents\Projetos\PINNeAPPle\examples\app\backend\chassis SLDPRT.SLDPRT")

obj = doc.Objects[0]

Mesh.export([obj], r"C:\Users\Yan\Documents\Projetos\PINNeAPPle\examples\app\backend\chassis.stl")
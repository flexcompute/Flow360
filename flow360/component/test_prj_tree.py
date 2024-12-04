import flow360 as fl
from flow360.examples import Airplane

# project = fl.Project.from_cloud("prj-7fbf25ba-72c8-44d4-be02-f2404f587749")

project = fl.Project.from_file(Airplane.geometry,
                               name="Python Project from Geometry")
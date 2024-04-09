from flow360 import MyCases
from flow360.environment import Env

env = Env.current.name

my_cases = MyCases()

for case in my_cases:
    if "UDD" in case.name:
        print("Downloading surfaces")
        try:
            case.results.download(surface=True, destination="./results")
        except Exception as e:
            print(f"Surfaces failed with {e}")

        print("Downloading volumes")
        try:
            case.results.download(volume=True, destination="./results")
        except Exception as e:
            print(f"Volumes failed with {e}")

        print("Downloading slices")
        try:
            case.results.download(slices=True, destination="./results")
        except Exception as e:
            print(f"Slices failed with {e}")

        print("Downloading iso")
        try:
            case.results.download(isosurfaces=True, destination="./results")
        except Exception as e:
            print(f"Iso failed with {e}")

        print("Downloading monitors")
        try:
            case.results.download(monitors=True, destination="./results")
        except Exception as e:
            print(f"Monitors failed with {e}")

        print("Downloading nonlinear residuals")
        try:
            case.results.download(nonlinear_residuals=True, destination="./results")
        except Exception as e:
            print(f"Nonlinear residuals failed with {e}")

        print("Downloading linear residuals")
        try:
            case.results.download(linear_residuals=True, destination="./results")
        except Exception as e:
            print(f"Linear residuals failed with {e}")

        print("Downloading CFL")
        try:
            case.results.download(cfl=True, destination="./results")
        except Exception as e:
            print(f"CFL failed with {e}")

        print("Downloading minmax")
        try:
            case.results.download(minmax_state=True, destination="./results")
        except Exception as e:
            print(f"Minmax failed with {e}")

        print("Downloading max residual")
        try:
            case.results.download(max_residual_location=True, destination="./results")
        except Exception as e:
            print(f"Max residual failed with {e}")

        print("Downloading surface forces")
        try:
            case.results.download(surface_forces=True, destination="./results")
        except Exception as e:
            print(f"Surface forces failed with {e}")

        print("Downloading total forces")
        try:
            case.results.download(total_forces=True, destination="./results")
        except Exception as e:
            print(f"Total forces failed with {e}")

        print("Downloading BET forces")
        try:
            case.results.download(bet_forces=True, destination="./results")
        except Exception as e:
            print(f"BET forces failed with {e}")

        print("Downloading actuator disks")
        try:
            case.results.download(actuator_disks=True, destination="./results")
        except Exception as e:
            print(f"Actuator disks failed with {e}")

        print("Downloading force distribution")
        try:
            case.results.download(force_distribution=True, destination="./results")
        except Exception as e:
            print(f"Force distribution failed with {e}")

        print("Downloading UDD")
        try:
            case.results.download(user_defined_dynamics=True, destination="./results")
        except Exception as e:
            print(f"UDD failed with {e}")

        print("Downloading aeroacoustic")
        try:
            case.results.download(aeroacoustics=True, destination="./results")
        except Exception as e:
            print(f"Aeroacoustic failed with {e}")

        print("Downloading surface heat")
        try:
            case.results.download(surface_heat_transfer=True, destination="./results")
        except Exception as e:
            print(f"Surface heat failed with {e}")
        break


print("Done")

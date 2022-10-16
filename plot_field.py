import numpy as np
from pprint import pprint
from matplotlib import pyplot as plt

class PlotField:

    def __init__(self,
        Nx: int, 
        Ny: int,
        Lx = float,
        Ly = float,
        particles = None
    ):

        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.particles = particles

        self._inititalize_coordinates()


    def _inititalize_coordinates(self):


        x = np.linspace(-self.Lx/2, self.Lx/2, self.Nx)
        y = np.linspace(-self.Ly/2, self.Ly/2, self.Ny)

        x, y = np.meshgrid(x,y)

        x = x.flatten()
        y = y.flatten()

        exterior_pts = {}
        interior_pts = {}
        all_int_idxs = []
        for i, p in enumerate(self.particles):

            r = np.sqrt((x - p["x"])**2 + (y - p["y"])**2)
            idxs = r <= p["r"]
            interior_pts[i] = {}
            interior_pts[i]["idxs"] = idxs
            interior_pts[i]["x"] = x[idxs]
            interior_pts[i]["y"] = y[idxs]

            all_int_idxs.append(idxs)

        ext_idxs = [not x for x in np.logical_or.reduce(all_int_idxs)]

        exterior_pts["idxs"] = ext_idxs
        exterior_pts["x"] = x[ext_idxs]
        exterior_pts["y"] = y[ext_idxs]


        self.x = x
        self.y = y
        self.exterior_pts = exterior_pts
        self.interior_pts = interior_pts


        '''
        plt.scatter(exterior_pts["x"], exterior_pts["y"],s=.5)
        plt.savefig("exterior.png")

        for k, v in interior_pts.items():

            plt.scatter(v["x"], v["y"], s=0.5)
            plt.savefig(f"interior_{k}.png")
        raise Exception
        '''


if __name__ == "__main__":

    obj = PlotField(128,128,8,8)





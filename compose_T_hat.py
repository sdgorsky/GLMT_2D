from scipy.special.cython_special import jv as J, hankel1 as H
import math
import cmath
import numpy as np
from pydantic import BaseModel, confloat
from matplotlib import pyplot as plt

from plot_field import PlotField

class SourcePlaneWave(BaseModel):
    wavelength: float
    theta: confloat(ge=0, le=2*math.pi)



    def calc_phiz(self, x, y):

        ko = 2*math.pi/self.wavelength

        xcoeff = 1j*ko*cmath.cos(self.theta)
        ycoeff = 1j*ko*cmath.sin(self.theta)

        return [cmath.exp(xcoeff*xi + ycoeff*yi) for xi, yi in zip(x, y)]


class Particle(BaseModel):
    x: float
    y: float
    #r: posfloat
    #epsr: complex = 1.0 + 1.0j


#class SimulationConfigs(BaseModel):
    #particles: list[Particle]
    #wavelength: posfloat
    #lmax: posint

    #def k(self, n: int = None) -> complex:
    #
    #    index = 1
    #    if isinstance(n, int) and n>=0:
    #        index = cmath.sqrt(self.particles[n].epsr)
    #
    #
    #    return index*2*math.pi/self.wavelength



class matrix_calculator:

    def __init__(self, particles, wavelength, Lmax):


        self.particles = particles
        self.wavelength = wavelength
        self.Lmax = Lmax

        self.source = SourcePlaneWave(wavelength=self.wavelength, theta=math.pi/4)

    def calculate_T_coeff(self):


        ko = 2*math.pi/self.wavelength

        Np = len(self.particles)
        Nell = 2*self.Lmax + 1

        s = np.zeros((Np,Nell), dtype=complex)
        #t = np.zeros((Np,Nell), dtype=complex)
        gamma = np.zeros((Np,Nell), dtype=complex)
        a0E = np.zeros((Np,Nell), dtype=complex)
        a0_hat = np.zeros((Np,Nell), dtype=complex)
        a = np.zeros((Np,Nell), dtype=complex)
        b = np.zeros((Np,Nell), dtype=complex)
        c = np.zeros((Np,Nell), dtype=complex)
        
        for n, p in enumerate(self.particles):

            kn = ko*np.sqrt(p["epsr"])

            kor = ko*p["r"]
            knr = kn*p["r"]
            xin = 1.0
            coeff = xin*kn/ko


            th = self.source.theta
            
            for ell in range(-self.Lmax, self.Lmax+1):

                

                gamma = coeff * Jp(ell, knr) / J(ell, knr)

                num = Jp(ell, kor) - gamma * J(ell, kor)
                den = Hp(ell, kor) - gamma * H(ell, kor)
                s[n][ell] = -num/den


                num = coeff*Hp(ell, knr) - gamma*H(ell, knr)
                den = Hp(ell, kor) - gamma*H(ell, kor)
                #t[i][ell] = num/den

                atmp = cmath.exp(1j*ko*(p["x"]*math.cos(th) + p["y"]*math.sin(th)))
                btmp = cmath.exp(-1j*ell*th)
                a0E[n][ell] = ((1j)**ell)*atmp*btmp

                a0_hat[n][ell] = s[n][ell]*a0E[n][ell] / J(ell, kor)


        T = np.identity(len(self.particles)*(2*self.Lmax+1), dtype=complex)

        Np = len(self.particles)
        Nell = 2*self.Lmax + 1
        for n in range(Np):
            for npr in range(Np):
                if n != npr:
                    for lidx, ell in enumerate(range(-self.Lmax, self.Lmax+1)):
                        for lidxp, ellp in enumerate(range(-self.Lmax, self.Lmax+1)):


                            i = (n-1)*Nell  + lidx
                            k = (npr-1)*Nell + lidxp


                            x = self.particles[n]["x"]
                            y = self.particles[n]["y"]
                            r = self.particles[n]["r"]

                            xp = self.particles[npr]["x"]
                            yp = self.particles[npr]["y"]
                            rp = self.particles[npr]["r"]

                            R = math.sqrt((x-xp)**2 + (y-yp)**2)
                            phi = math.atan2(yp-y, xp-x)
                            phase = cmath.exp(1j*(ellp-ell)*phi)

                            T[k][i] = -phase*H(ell-ellp, ko*R)*s[n][ell]*Jp(ellp, ko*rp)/J(ell, ko*r)


        Tinv = np.linalg.inv(T)

        bhat = np.matmul(Tinv, a0_hat.reshape(-1))
        bhat = bhat.reshape(Np,Nell)

        for n in range(len(self.particles)):
            for ell in range(-self.Lmax,self.Lmax+1):

                r = self.particles[n]["r"]
                kn = ko*np.sqrt(self.particles[n]["epsr"])

                b[n][ell] = bhat[n][ell] * J(ell, ko*r)
                a[n][ell] = b[n][ell] / s[n][ell]

                c[n][ell] = (a[n][ell]*J(ell, ko*r) + b[n][ell]*H(ell, ko*r)) / J(ell, kn*r)


        print(c)
        self.b = b
        self.c = c


    def calc_fields(self):

        coords = PlotField(Nx=128,Ny=128,Lx=5,Ly=5,particles=self.particles)

        src = self.calc_phiz_src()
        sca = self.calc_phiz_sca()
        int = self.calc_phiz_int()


        x = coords.x
        y = coords.y


        phiz = np.zeros(np.shape(x), dtype=complex)

        phiz[src[0]] += src[1]
        phiz[sca[0]] += sca[1]

        for field in int:
            phiz[field[0]] += field[1]

        plt.clf()
        plt.scatter(x, y, c=np.real(phiz))
        plt.colorbar()
        plt.savefig("phiz.png")

    
    def calc_phiz_src(self):

        print("Calculating phi_z_src")

        coords = PlotField(Nx=128,Ny=128,Lx=5,Ly=5,particles=self.particles)

        x = coords.exterior_pts["x"]
        y = coords.exterior_pts["y"]

        phiz_src = self.source.calc_phiz(x,y)

        plt.clf()
        plt.scatter(x, y, c=np.real(phiz_src))
        plt.colorbar()
        plt.savefig("phiz_src.png")

        return (coords.exterior_pts["idxs"], phiz_src)

    def calc_phiz_sca(self):

        print("Calculating phi_z_sca")

        coords = PlotField(Nx=128,Ny=128,Lx=5,Ly=5,particles=self.particles)

        x = coords.exterior_pts["x"]
        y = coords.exterior_pts["y"]

        phiz_sca = np.zeros(np.shape(x),dtype=complex)

        b = self.b
        ko = 2*math.pi/self.wavelength

        for n, p in enumerate(self.particles):

            rhon = np.sqrt((x-p["x"])**2 + (y-p["y"])**2)
            thn = np.arctan2(y-p["y"], x-p["x"])

            for ell in range(-self.Lmax, self.Lmax+1):


                phiz_sca += [b[n][ell]*H(ell, ko*rho)*cmath.exp(1j*ell*th) for (rho,th) in zip(rhon, thn)]

        
        plt.clf()
        plt.scatter(x, y, c=np.real(phiz_sca))
        plt.colorbar()
        plt.savefig("phiz_sca.png")

        return (coords.exterior_pts["idxs"], phiz_sca)


    def calc_phiz_int(self):

        print("Calculating phi_z_int")

        coords = PlotField(Nx=128,Ny=128,Lx=5,Ly=5,particles=self.particles)

        ko = 2*math.pi/self.wavelength

        c = self.c

        phiz_int_all = []

        for n, p in enumerate(self.particles):

            x = coords.interior_pts[n]["x"]
            y = coords.interior_pts[n]["y"]

            phiz_cur = np.zeros(np.shape(x),dtype=complex)
            rhon = np.sqrt((x-p["x"])**2 + (y-p["y"])**2)
            thn = np.arctan2(y-p["y"], x-p["x"])
            kn = ko*np.sqrt(p["epsr"])


            for ell in range(-self.Lmax, self.Lmax+1):
               phiz_cur += [c[n][ell]*J(ell, kn*rho)*cmath.exp(1j*ell*th) for (rho, th) in zip(rhon, thn)]

            
            phiz_int_all.append((coords.interior_pts[n]["idxs"], phiz_cur))
            plt.clf()
            plt.scatter(x, y, c=np.real(phiz_cur))
            plt.colorbar()
            plt.savefig(f"phiz_int_{n}.png")

        return phiz_int_all


def Jp(nu, x):
    """ First derivative of Bessel-J"""
    return x*(J(nu-1,x) - J(nu+1,x))/2

def Hp(nu, x):
    """ First derivative of Bessel-H(1)"""
    return x*(H(nu-1,x) - H(nu+1,x))/2



if __name__ == "__main__":
    print("Running")

    particles = [
        {
            "x": -1.5,
            "y": 0.5,
            "r": 0.5,
            "epsr": 2.2
        },
        {
            "x": 0.6,
            "y": -0.25,
            "r": 0.25,
            "n": 1.5,
            "epsr": 5
        },
        {
            "x": 0.2,
            "y": 2.0,
            "r": 0.25,
            "n": 1.5,
            "epsr": 5.5
        }

    ]


    obj = matrix_calculator(
        particles = particles,
        wavelength = 0.5,
        Lmax=20


    )
    
    obj.calculate_T_coeff()
    obj.calc_fields()
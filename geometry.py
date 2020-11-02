import numpy as np
from sklearn.utils import shuffle
import time

class Rectangle:
    def __init__ (self, nx, nz, nt, ox, oz, ot, dx, dz, dt):
        # . . two spatial and one temporal dimensions
        # . . grid size
        # . . while domain: interior + boundaries
        self.nx = nx
        self.nz = nz
        self.nt = nt
        # . . interior domain: ecluding boundaries
        self.nxint = nx - 2
        self.nzint = nz - 2
        # . . interior points of the temporal grid: exclude the initial condition
        self.ntint = nt - 1
        # .  grid spacings
        self.dx = dx
        self.dz = dz
        self.dt = dt

        # . . compute the boundaries
        # . . left boundary
        self.lb = ox
        # . . right boundary
        self.rb = ox + (nx-1)*dx
        # . . top boundary
        self.tb = oz
        # . . bottom boundary
        self.bb = oz + (nz-1)*dz
        # . . starting time (t0)
        self.t0 = ot
        # . . end time
        self.tN = ot + (nt-1)*dt 

        # . . create the grids
        # . . spatial grid
        self.x = np.reshape(np.linspace(self.lb, self.rb, self.nx), (self.nx, 1))
        self.z = np.reshape(np.linspace(self.tb, self.bb, self.nz), (self.nz, 1))
        # . . temporal grid
        self.t    = np.reshape(np.linspace(self.t0, self.tN, self.nt), (self.nt, 1))        

    def points(self, shuffle=False):
        # . . total number of spatial grid points
        nxz    = self.nx*self.nz        
        # . . total number of interior grid points
        nxzint = self.nxint*self.nzint
        # . . left + right + top + bottom: counts corners twice
        nxzbnd = 2*self.nx + 2*self.nz        

        # . . interior grid
        # . . spatial grid
        xint = np.reshape(np.linspace(self.lb+self.dx, self.rb-self.dx, (self.nxint)), ((self.nxint), 1))
        zint = np.reshape(np.linspace(self.tb+self.dz, self.bb-self.dz, (self.nzint)), ((self.nzint), 1))
        # . . temporal grid
        tint = np.reshape(np.linspace(self.dt, self.tN, self.ntint), (self.ntint, 1))

        # . . interior points
        xx, zz = np.meshgrid(xint, zint) 
        
        xx = np.tile(np.reshape(xx, (nxzint, 1)), (self.ntint, 1))        
        zz = np.tile(np.reshape(zz, (nxzint, 1)), (self.ntint, 1))  
            
        tt = np.repeat(tint, nxzint).reshape((self.ntint*nxzint, 1))

        # . . the interior grid
        interior = np.concatenate((tt, zz, xx), axis=1)

        # . . boundary points
        # . . top boundary
        zt = np.reshape(np.repeat(self.tb, self.nx), (self.nx, 1))
        top = np.concatenate((zt, self.x), axis=1)
        # . . left boundary        
        xl = np.reshape(np.repeat(self.lb, self.nz), (self.nz, 1))
        left = np.concatenate((self.z, xl), axis=1)
        # . . bottom boundary
        zb = np.reshape(np.repeat(self.bb, self.nx), (self.nx, 1))
        bottom = np.concatenate((zb, self.x), axis=1)
        # . . right boundary
        xr = np.reshape(np.repeat(self.rb, self.nz), (self.nz, 1))
        right = np.concatenate((self.z, xr), axis=1)

        # . . combine boundary point counterclockwise: [top; left; bottom; right]
        sides = np.concatenate((top,left,bottom,right),axis=0)

        # . . new time grid for boundary points
        tt = np.repeat(tint, nxzbnd).reshape((self.ntint*nxzbnd, 1))

        # . . boundary points for all time steps
        sides = np.tile(sides, (self.ntint, 1)) 
    
        # . . the boundary grid
        boundary = np.concatenate((tt, sides), axis=1)

        # . . initial conditions
        t0 = np.reshape(np.repeat(self.t0, nxz), (nxz, 1))
        
        # . . grid points
        xx, zz = np.meshgrid(self.x, self.z) 
        
        xx = np.reshape(xx, (nxz, 1))
        zz = np.reshape(zz, (nxz, 1))

        # . . combine the coordinates for the initial conditions
        initial = np.concatenate((t0, zz, xx),axis=1)

        # . . shuffle the spatio-temporal grid for training
        if (shuffle):
            #np.random.shuffle(grid)
            interior = shuffle(interior, random_state=int(time.time()))
            boundary = shuffle(boundary, random_state=int(time.time()))
            initial  = shuffle(initial , random_state=int(time.time()))
        
        # . . return the spatiotemporal grid
        return interior, boundary, initial


    def spatiotemporal(self, shuffle=False):
        # . . total number of spatial grid points
        nxz  = self.nx*self.nz

        xx, zz = np.meshgrid(self.x, self.z) 
        
        xx = np.tile(np.reshape(xx, (nxz, 1)), (self.nt, 1))        
        zz = np.tile(np.reshape(zz, (nxz, 1)), (self.nt, 1))  
            
        tt = np.repeat(self.t, nxz).reshape((self.nt*nxz, 1))

        grid = np.concatenate((tt, zz, xx), axis=1)

        # . . shuffle the spatio-temporal grid for training
        if (shuffle):
            #np.random.shuffle(grid)
            grid = shuffle(grid, random_state=int(time.time()))
        
        # . . return the spatiotemporal grid
        return grid

    def meshattime(self, time):
        # . . total number of spatial grid points
        nxz  = self.nx*self.nz

        xx, zz = np.meshgrid(self.x, self.z) 
        
        xx = np.reshape(xx, (nxz, 1))        
        zz = np.reshape(zz, (nxz, 1))  
            
        tt = np.repeat(time, nxz).reshape((nxz, 1))

        grid = np.concatenate((tt, zz, xx), axis=1)
        
        # . . return the spatiotemporal grid
        return grid        
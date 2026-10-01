"""Planar native-surface geometry and an explicitly defined P1 heat adapter.

This is a lumped-mass surface diffusion approximation, not the original 3D FEM
solver. Outer boundary temperature excess is fixed to zero during propagation.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import expm_multiply


@dataclass
class SurfaceMesh:
    points: np.ndarray
    triangles: np.ndarray
    areas: np.ndarray
    weights: np.ndarray
    boundary: np.ndarray
    stiffness: object


def surface_mesh(points, triangles):
    xy=np.asarray(points,dtype=float)
    tri=np.asarray(triangles,dtype=int).copy()
    if xy.ndim!=2 or xy.shape[1]!=2 or tri.ndim!=2 or tri.shape[1]!=3:
        raise ValueError('Expected planar points and triangular surface connectivity')
    if tri.min()<0 or tri.max()>=len(xy): raise ValueError('Invalid node index')
    p=xy[tri]
    determinant=(p[:,1,0]-p[:,0,0])*(p[:,2,1]-p[:,0,1])-(p[:,2,0]-p[:,0,0])*(p[:,1,1]-p[:,0,1])
    if np.any(abs(determinant)<1e-20): raise ValueError('Degenerate surface triangle')
    negative=determinant<0
    tri[negative]=tri[negative][:,[0,2,1]]
    p=xy[tri]; det=abs(determinant); area=det/2
    gx=np.column_stack([p[:,1,1]-p[:,2,1],p[:,2,1]-p[:,0,1],p[:,0,1]-p[:,1,1]])/det[:,None]
    gy=np.column_stack([p[:,2,0]-p[:,1,0],p[:,0,0]-p[:,2,0],p[:,1,0]-p[:,0,0]])/det[:,None]
    local=area[:,None,None]*(gx[:,:,None]*gx[:,None,:]+gy[:,:,None]*gy[:,None,:])
    rows=np.broadcast_to(tri[:,:,None],local.shape).ravel()
    cols=np.broadcast_to(tri[:,None,:],local.shape).ravel()
    k=coo_matrix((local.ravel(),(rows,cols)),shape=(len(xy),len(xy))).tocsr()
    mass=np.bincount(tri.ravel(),weights=np.repeat(area/3,3),minlength=len(xy))
    if np.any(mass<=0): raise ValueError('Unconnected surface nodes')
    edges=np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1)
    edges,counts=np.unique(edges,axis=0,return_counts=True)
    if np.any(counts>2): raise ValueError('Non-manifold surface')
    boundary=np.zeros(len(xy),bool); boundary[np.unique(edges[counts==1])]=True
    return SurfaceMesh(xy,tri,area,mass,boundary,k)


def extract_surface(trajectory):
    ids=trajectory.surface_node_indices(); xyz=trajectory.geometry()
    cells=trajectory.topology()
    if cells.shape[1]!=4: raise ValueError('This extractor requires tetrahedra')
    lookup=np.full(len(xyz),-1,int); lookup[ids]=np.arange(len(ids))
    mapped=lookup[cells]; selected=mapped[np.sum(mapped>=0,axis=1)==3]
    tri=np.array([r[r>=0] for r in selected])
    tri=np.unique(np.sort(tri,axis=1),axis=0)
    return surface_mesh(xyz[ids,:2],tri)


def propagate(mesh, values, *, dt, diffusivity, cooling_rate, block_size=64):
    """Return exp(dt L) values on nodes, applying homogeneous Dirichlet excess."""
    if min(dt,diffusivity,cooling_rate)<0: raise ValueError('Negative physical parameter')
    a=np.asarray(values,dtype=float)
    if a.shape[-1]!=len(mesh.points): raise ValueError('Last axis must be native nodes')
    flat=a.reshape(-1,len(mesh.points)); out=np.zeros_like(flat)
    interior=np.flatnonzero(~mesh.boundary)
    if not len(interior): return out.reshape(a.shape)
    k=mesh.stiffness[interior][:,interior]
    generator=-diffusivity*diags(1/mesh.weights[interior])@k-cooling_rate*diags(np.ones(len(interior)))
    step=dt*generator
    for start in range(0,len(flat),block_size):
        out[start:start+block_size,interior]=expm_multiply(step,flat[start:start+block_size,interior].T,
            traceA=float(step.diagonal().sum())).T
    return out.reshape(a.shape)


def physics_forecast(mesh, previous, flux, *, ambient, dt, diffusivity, cooling_rate, source_coupling, source_flux_threshold):
    background=propagate(mesh,np.maximum(np.asarray(previous)-ambient,0),dt=dt,diffusivity=diffusivity,cooling_rate=cooling_rate)
    source=np.where(np.asarray(flux)>=source_flux_threshold,flux,0.)*source_coupling*dt
    return ambient+background+source


def hottest_area_weights(truth,weights,fraction=.01):
    """Fractional final weight makes the selected area exactly the target fraction."""
    if not 0<fraction<=1: raise ValueError('Invalid fraction')
    order=np.argsort(-np.asarray(truth),kind='stable')
    w=np.asarray(weights,dtype=float)
    if np.any(w<0) or w.sum()<=0: raise ValueError('Invalid area weights')
    remaining=fraction*w.sum(); result=np.zeros_like(w)
    for i in order:
        take=min(w[i],remaining); result[i]=take; remaining-=take
        if remaining<=0: break
    return result


def quadrature(mesh, refinement=0):
    """Degree-two triangle rule, optionally on uniformly subdivided triangles."""
    if refinement not in (0,1,2,3):
        raise ValueError('Quadrature refinement must be 0, 1, 2 or 3')
    bary=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    children=np.eye(3)[None]
    for _ in range(refinement):
        a,b,c=children[:,0],children[:,1],children[:,2]
        ab,bc,ca=(a+b)/2,(b+c)/2,(c+a)/2
        children=np.concatenate([np.stack(v,axis=1) for v in
            [(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)]])
    bary=np.einsum('qv,tvd->tqd',bary,children).reshape(-1,3)
    points=np.einsum('qv,tvd->tqd',bary,mesh.points[mesh.triangles]).reshape(-1,2)
    weights=np.repeat(mesh.areas/len(bary),len(bary))
    return points,weights,bary

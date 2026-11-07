"""Reproduces all numerical results in "Pendular Urban Locomotion with Tethered Anchors". Requires numpy, scipy."""
import numpy as np
from scipy.integrate import solve_ivp
g=9.81; m=85.0; rho=1.2
def swing(L0=30.0, th0=np.radians(90), v0=0.0, CdA=0.5, reel=None, th_rel=None, tmax=20):
    # state: theta (from vertical, +ve = forward side), omega ; anchor at origin, body at (L sin th, -L cos th)
    # reel: function(theta)-> (L, Ldot_per_omega) ; we integrate L as state for simplicity
    def Lfun(th):
        if reel is None: return L0,0.0
        return reel(th)
    def rhs(t,y):
        th,w,L=y
        Ld_dth = Lfun(th)[1]; Ld = Ld_dth*w
        v_t = L*w  # tangential
        v = np.hypot(v_t,Ld)
        D = 0.5*rho*CdA*v*v
        # tangential drag component
        Dt = -D*(v_t/v) if v>1e-9 else 0
        # Lagrangian eq for variable length: d/dt(m L^2 w) = -m g L sin th + Dt*L
        # m L^2 wdot + 2 m L Ld w = ...
        wdot = (-m*g*L*np.sin(th) + Dt*L - 2*m*L*Ld*w)/(m*L*L)
        return [w,wdot,Ld]
    # start: th0 negative (behind), moving forward
    w0 = v0/L0
    ev = lambda t,y: y[0]-th_rel if th_rel is not None else 1
    ev.terminal=True; ev.direction=1
    sol=solve_ivp(rhs,[0,tmax],[-th0,w0,L0],events=ev,max_step=0.002,rtol=1e-8)
    return sol
def tension(sol,CdA=0.5,reel_acc=None):
    th,w,L=sol.y
    # radial: T - mg cos th = m L w^2  (ignoring Lddot, drag radial ~0 for rigid)
    return m*g*np.cos(th)+m*L*w**2
out={}
# 1. peak tension vs start angle, vacuum and drag
rows=[]
for deg in [30,45,60,75,90]:
    s0=swing(th0=np.radians(deg),CdA=0.0,th_rel=0.0)
    s1=swing(th0=np.radians(deg),CdA=0.5,th_rel=0.0)
    T0=tension(s0).max()/(m*g); T1=tension(s1).max()/(m*g)
    v0=30*abs(s0.y[1,-1]); v1=30*abs(s1.y[1,-1])
    rows.append((deg,round(v0,1),round(v1,1),round(T0,2),round(T1,2)))
print("start_deg, vbottom_vac, vbottom_drag, Tmax/mg vac, Tmax/mg drag");[print(r) for r in rows]
# 2. release angle sweep: start horizontal rope (90deg) from rest, L=30, anchor at height
def flight(x0,y0,vx,vy,CdA,ytarget):
    def f(t,s):
        x,y,vx,vy=s; v=np.hypot(vx,vy); k=0.5*rho*CdA*v/m
        return [vx,vy,-k*vx,-g-k*vy]
    ev=lambda t,s: s[1]-ytarget; ev.terminal=True; ev.direction=-1
    r=solve_ivp(f,[0,30],[x0,y0,vx,vy],events=ev,max_step=0.005,rtol=1e-8)
    return r
print("\nrelease sweep (start 90deg, L=30). metric: horizontal distance from start point until body falls back to start altitude (y=0 of anchor) and to bottom-of-arc altitude")
best={}
for CdA in [0.0,0.5]:
  res=[]
  for rd in range(0,85,5):
    s=swing(th0=np.radians(90),CdA=CdA,th_rel=np.radians(rd)) if rd>0 else swing(th0=np.radians(90),CdA=CdA,th_rel=1e-6)
    th,w,L=s.y[:,-1]
    x=L*np.sin(th); y=-L*np.cos(th); vt=L*w
    vx=vt*np.cos(th); vy=vt*np.sin(th)
    xstart=-30.0
    # land at altitude 10 m below release-of-start (i.e., y=-10 relative to anchor) -> represents next anchor chain needing net drop
    r=flight(x,y,vx,vy,CdA,-30.0)  # back to bottom-of-arc altitude
    dist=r.y[0,-1]-xstart
    apex=r.y[1].max()
    res.append((rd,round(np.hypot(vx,vy),1),round(dist,1),round(apex,1)))
  print("CdA",CdA); [print(q) for q in res]
  best[CdA]=max(res,key=lambda q:q[2])
print("best",best)
# 3. reel-in pumping: shorten from 30 to 26 m over +-15deg around bottom
def reel(th):
    a=np.radians(15)
    if th<-a: return 30.0,0.0
    if th>a: return 26.0,0.0
    # linear in theta
    return 30.0-4.0*(th+a)/(2*a), -4.0/(2*a)
for CdA in [0.0,0.5]:
    s=swing(th0=np.radians(90),CdA=CdA,reel=reel,th_rel=np.radians(40))
    sN=swing(th0=np.radians(90),CdA=CdA,th_rel=np.radians(40))
    th,w,L=s.y
    T=m*g*np.cos(th)+m*L*w**2
    Ld=np.gradient(L,s.t)
    P=(-T*Ld).max()
    def E(sol):
        th,w,L=sol.y[:,-1]; return 0.5*m*(L*w)**2 + m*g*(-L*np.cos(th)) 
    print(f"\nreel CdA={CdA}: v at 40deg reel {L[-1]*w[-1]:.1f} vs no reel {sN.y[2,-1]*sN.y[1,-1]:.1f}; Tmax/mg {T.max()/(m*g):.2f}; peak winch power {P/1000:.1f} kW; work {np.trapezoid(-T*Ld,s.t)/1000:.1f} kJ; dE {(E(s)-E(sN))/1000:.1f} kJ")
# 4. elastic catch: slack fall h, rope length L, EA
for EA,name in [(20e3,'dynamic kernmantle ~20 kN/strain'),(60e3,'semi-static ~60kN'),(300e3,'UHMWPE ~300 kN')]:
    for f in [0.3,1.0]:
        F=m*g+np.sqrt((m*g)**2+2*m*g*EA*f)
        print(name,"fall factor",f,"Fmax kN",round(F/1000,2),"g",round(F/(m*g),1))

# --- 1-degree refinement of optimal release angle ---
def flight(x0,y0,vx,vy,CdA,yt):
    def f(t,s):
        x,y,vx,vy=s; v=np.hypot(vx,vy); k=0.5*rho*CdA*v/m
        return [vx,vy,-k*vx,-g-k*vy]
    ev=lambda t,s: s[1]-yt; ev.terminal=True; ev.direction=-1
    return solve_ivp(f,[0,30],[x0,y0,vx,vy],events=ev,max_step=0.005,rtol=1e-8)
for CdA in [0.0,0.5]:
  best=(0,0)
  for rd in np.arange(30,50,1.0):
    s=swing(th0=np.radians(90),CdA=CdA,th_rel=np.radians(rd))
    th,w,L=s.y[:,-1]; vt=L*w
    r=flight(L*np.sin(th),-L*np.cos(th),vt*np.cos(th),vt*np.sin(th),CdA,-30)
    d=r.y[0,-1]+30
    if d>best[1]: best=(rd,d,r.t[-1]+s.t[-1])
  print(CdA,best)
